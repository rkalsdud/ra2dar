"""Retina-4SN viewer — Three.js + WebSocket 단일 파일 버전.

배경: Dash + Plotly 가 Tailscale 원격 환경에서 너무 느려서(3초+ 갱신)
      대안으로 가볍게 Three.js 직접 렌더링.

흐름:
    Retina (user_module) ── TCP 29173 ──► viewer_three.py ──► Flask + WebSocket
                                                                 │
                                                                 ▼
                                            브라우저 (Three.js + HTML/CSS)

차별점 (vs viewer_go.py / Plotly Dash):
    · WebSocket push: 서버가 매 REFRESH_MS 마다 클라이언트로 frame push.
      Dash 의 long-polling/HTTP POST 왕복 RTT 없음.
    · Three.js BufferAttribute 직접 갱신: 점 좌표만 in-place update.
      Plotly 의 figure 통째 재전송 없음 → 대역 1/10 수준.
    · 단일 파일: viewer_three.py 하나 (HTML/JS 임베드).
      requirements: flask, flask-sock

실행:
    python viewer_three.py --retina-host 192.168.0.100
    # 브라우저 http://<this-host>:8050/

의존성:
    pip install flask flask-sock
"""
from __future__ import annotations

import argparse
import bisect
import glob
import json
import os
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, List, Optional

try:
    from flask import Flask, abort, jsonify, request, send_file
    from flask_sock import Sock
except ImportError:
    print("Required packages 누락. 다음을 실행해주세요:")
    print("  pip install flask flask-sock")
    raise SystemExit(1)

import os
# 오탐 보고 저장 위치 — viewer 실행 경로 기준 reports/.
REPORTS_DIR = "reports"
# 보고 시점 기준으로 saved 되는 frame buffer (롤링, ~30초 @ 20Hz).
_frame_history: deque = deque(maxlen=600)


# ============ JPG 좌측 패널 (--jpg-dir 옵션) ============

_jpg_ts_sorted: list = []
_jpg_by_ts: dict = {}
_HAS_JPG = False


def build_jpg_index(jpg_dir: str) -> int:
    """matched_dataset_<N>/{ts}.jpg 들을 인덱싱."""
    global _jpg_ts_sorted, _jpg_by_ts
    out = {}
    for p in sorted(glob.glob(os.path.join(jpg_dir, "*.jpg"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            out[float(stem)] = p
        except ValueError:
            continue
    _jpg_by_ts = out
    _jpg_ts_sorted = sorted(out.keys())
    if _jpg_ts_sorted:
        print(f"[jpg] {len(_jpg_ts_sorted)} files indexed "
              f"({_jpg_ts_sorted[0]:.3f} ~ {_jpg_ts_sorted[-1]:.3f})")
    else:
        print(f"[jpg] WARN: {jpg_dir} 에 *.jpg 없음")
    return len(_jpg_ts_sorted)


def find_jpg_basename(ts: float, max_diff_s: float = 1.0) -> Optional[str]:
    """가장 가까운 JPG basename (Flask route 가 /jpg/<basename> 으로 서빙)."""
    if not _jpg_ts_sorted or ts <= 0:
        return None
    i = bisect.bisect_left(_jpg_ts_sorted, ts)
    candidates = []
    if i < len(_jpg_ts_sorted): candidates.append(_jpg_ts_sorted[i])
    if i > 0: candidates.append(_jpg_ts_sorted[i - 1])
    if not candidates:
        return None
    best = min(candidates, key=lambda c: abs(c - ts))
    if abs(best - ts) > max_diff_s:
        return None
    p = _jpg_by_ts.get(best)
    return os.path.basename(p) if p else None


# ============ FrameData ============

@dataclass
class FrameData:
    frame: int = 0
    points: List[List[float]] = field(default_factory=list)   # [[x,y,z,V,P,track_id], ...]
    tracks: List[dict] = field(default_factory=list)          # cascade sequence
    received_at: float = 0.0
    timestamp: float = 0.0   # 원본 캡처 시각 — JSON 의 "T" 키 보존 (JPG 매칭용)


_latest = FrameData()
_lock = threading.Lock()

# 알람 상태 (전역, viewer_go.py 와 동등).
_alarm = {
    "events": 0,
    "last_seen": 0.0,
    "was_present": False,
    # per-track alarm tracking: track_id → last_seen_time.
    # 새로운 track_id 가 등장하면 알람. 같은 id 가 NOTIF_REARM_S 동안 미검출 시 제거.
    "active_ids": {},
}


# ============ 수신 thread: TCP (live) ============

def tcp_loop(host: str, port: int) -> None:
    """Retina JSON publisher 에 영구 접속. 끊기면 backoff 재연결."""
    global _latest
    backoff = 1.0
    while True:
        try:
            print(f"[tcp] connecting to {host}:{port} ...")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(10.0)
                s.connect((host, port))
                s.settimeout(None)
                print("[tcp] connected")
                backoff = 1.0
                rx_count = 0
                last_report = time.time()
                with s.makefile("r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            d = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        fd = FrameData(
                            frame=int(d.get("frame", 0)),
                            points=d.get("points", []) or [],
                            tracks=d.get("tracks", []) or [],
                            received_at=time.time(),
                            timestamp=float(d.get("T", 0.0)),
                        )
                        with _lock:
                            _latest = fd
                        # rolling buffer — 오탐 보고 시 최근 ~30초 frame snapshot 사용.
                        _frame_history.append({
                            "f": fd.frame,
                            "T": fd.timestamp,
                            "pts": fd.points,
                            "tracks": fd.tracks,
                            "ts": fd.received_at,
                        })
                        rx_count += 1
                        now = time.time()
                        if now - last_report >= 2.0:
                            print(f"[rx] {rx_count / (now - last_report):.1f} f/s "
                                  f"(frame={fd.frame}, pts={len(fd.points)})")
                            rx_count = 0
                            last_report = now
                print("[tcp] stream ended, reconnecting")
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            print(f"[tcp] error: {e}; retry in {backoff:.1f}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 10.0)


# ============ 데모 (UI 확인용) ============

def demo_loop(fps: int) -> None:
    """장비 없이 합성 프레임 — 사람 트랙이 6초 주기로 등장/퇴장."""
    global _latest
    import math
    import random
    dt = 1.0 / max(fps, 1)
    f = 0
    while True:
        f += 1
        t = f * dt
        present = (int(t) % 6) < 3
        px = 1.2 * math.sin(t * 0.7)
        py = 2.5 + 0.5 * math.sin(t * 0.4)
        pz = 0.0
        points = [[random.uniform(-3, 3), random.uniform(0, 6),
                   random.uniform(-1.5, 1.0),
                   random.uniform(-1, 1), random.uniform(0, 1), -1]
                  for _ in range(120)]
        tracks: list[dict] = []
        if present:
            points += [[px + random.uniform(-0.3, 0.3),
                        py + random.uniform(-0.3, 0.3),
                        pz + random.uniform(-0.5, 0.5),
                        random.uniform(-1, 1), random.uniform(0, 1), 0]
                       for _ in range(40)]
            hp = 0.85 + 0.1 * math.sin(t * 3)
            tracks.append({
                "id": 1, "human_prob": hp,
                "bbox": [px - 0.4, px + 0.4, py - 0.4, py + 0.4, pz - 0.8, pz + 0.8],
            })
        with _lock:
            _latest = FrameData(frame=f, points=points, tracks=tracks,
                                received_at=time.time())
        _frame_history.append({
            "f": f, "T": 0.0, "pts": points, "tracks": tracks,
            "ts": time.time(),
        })
        time.sleep(dt)


# ============ 사람 검출 ============

# 알림/active 게이팅 — cascade human_prob 가 이 값 이상일 때만 사람으로.
# 0.5 는 학습 PR 커브 첫 P=1.0 지점이지만 라이브에선 falses 가 종종 흘러나옴.
# 0.7 = box 색 초록 임계와 일치 — UI 전체 게이팅 일관.
HUMAN_THRESHOLD = 0.7

# 같은 사람이 트래킹 일시 끊김으로 알림 여러 번 뜨는 것 방지 debounce.
# 사람이 "사라진" 으로 간주되려면 NOTIF_REARM_S 동안 연속 미검출이어야.
# 짧은 트래킹 dropout (1~3프레임) 은 흡수.
NOTIF_REARM_S = 8.0


def person_in_frame(fd: FrameData, threshold: float):
    """현재 프레임에서 사람으로 본 트랙. (있음?, 최대 human_prob, 사람 수)."""
    best = 0.0
    count = 0
    for t in fd.tracks:
        hp = float(t.get("human_prob", 0.0))
        if hp >= threshold:
            count += 1
            if hp > best:
                best = hp
    return count > 0, best, count


# ── 작은 ghost cluster 필터 ───────────────────────────────────────
# 라이브에서 멀쩡한 사람 근처에 작은 가짜 cluster 가 형성되는 경우 다수.
# 사람 근처 multi-path 반사 / 노이즈 점이 우연히 cluster 임계 (10점) 넘어 박스 생성.
# 이런 ghost 는 박스 절대 크기가 매우 작음 (모든 축 < 0.3m).
# 모델은 근처 사람의 시퀀스 패턴과 비슷한 doppler 받아 높은 human_prob 출력 → 오탐.
# → bbox 의 최대 축 길이가 임계 미만이면 비사람으로 demote (human_prob = 0).
#   사람의 가장 낮은 자세 (포복) 도 어느 한 축은 0.4m+ 이라 false negative 위험 작음.
MIN_BOX_EXTENT_M = 0.3


def _box_too_small(t: dict) -> bool:
    """track 의 bbox 가 사람이라기엔 너무 작은지 판정."""
    bbox = t.get("bbox")
    if not bbox or len(bbox) < 6:
        return True
    dx = float(bbox[1]) - float(bbox[0])
    dy = float(bbox[3]) - float(bbox[2])
    dz = float(bbox[5]) - float(bbox[4])
    return max(dx, dy, dz) < MIN_BOX_EXTENT_M


def _apply_size_filter(tracks: list) -> list:
    """작은 박스는 사람으로 인정 안 함 — human_prob 만 0 으로 demote, 박스는 유지.
    원본 tracks 는 mutate 안 함 (copy)."""
    out = []
    for t in tracks:
        if _box_too_small(t):
            t2 = dict(t)
            t2["human_prob"] = 0.0
            out.append(t2)
        else:
            out.append(t)
    return out


# ============ Flask + WebSocket ============

app = Flask(__name__)
sock = Sock(app)

REFRESH_MS = 100      # WebSocket push 주기 (default 10 Hz)
MAX_POINTS = 2000     # send 시 점 수 상한 (초과 시 stride subsample)


@app.route("/")
def index():
    # _HAS_JPG 를 클라이언트 전달 — JS 가 layout 분기.
    return INDEX_HTML.replace("__HAS_JPG__", "true" if _HAS_JPG else "false")


@app.route("/report", methods=["POST"])
def report_false_alarm():
    """클라이언트가 오탐 보고 시 호출. 최근 frame buffer + 메타 저장.
    저장 위치: reports/<timestamp>_<label>/
      - metadata.json : label, note, 시각, frame 카운트 등
      - frames.jsonl  : 매 line 한 frame JSON (롤링 버퍼 전체 dump)
    """
    data = request.get_json(silent=True) or {}
    label = str(data.get("label", "unknown"))[:40]
    # 파일명 안전 문자만 (영문/숫자/하이픈/언더스코어).
    safe_label = ''.join(c if c.isalnum() or c in '-_' else '_' for c in label)
    note = str(data.get("note", ""))[:1000]
    snapshot = list(_frame_history)
    if not snapshot:
        return jsonify({"ok": False, "msg": "no frames buffered"}), 400
    os.makedirs(REPORTS_DIR, exist_ok=True)
    ts_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    rid = f"{ts_str}_{safe_label}"
    rdir = os.path.join(REPORTS_DIR, rid)
    if os.path.exists(rdir):
        rid = f"{rid}_{int(time.time() * 1000) % 10000}"
        rdir = os.path.join(REPORTS_DIR, rid)
    os.makedirs(rdir, exist_ok=True)
    meta = {
        "report_id": rid,
        "label": label,
        "note": note,
        "reported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "frame_count": len(snapshot),
        "first_frame": snapshot[0].get("f"),
        "last_frame": snapshot[-1].get("f"),
        "first_ts": snapshot[0].get("ts"),
        "last_ts": snapshot[-1].get("ts"),
        "duration_s": round((snapshot[-1].get("ts", 0) - snapshot[0].get("ts", 0)), 3),
    }
    try:
        with open(os.path.join(rdir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        with open(os.path.join(rdir, "frames.jsonl"), "w", encoding="utf-8") as f:
            for fr in snapshot:
                f.write(json.dumps(fr) + "\n")
    except OSError as e:
        print(f"[report] save failed: {e}")
        return jsonify({"ok": False, "msg": str(e)}), 500
    print(f"[report] saved {rid} ({len(snapshot)} frames)")
    return jsonify({"ok": True, "report_id": rid, "frame_count": len(snapshot)})


@sock.route("/ws")
def ws_handler(ws):
    """클라이언트당 별도 스레드. _latest 가 갱신될 때만 push."""
    last_frame_sent = -1
    try:
        while True:
            with _lock:
                fd = _latest
            now = time.time()

            # ── per-track alarm tracking ────────────────────────────
            # 각 트랙 id 가 처음 등장할 때마다 알람 1번.
            # 같은 id 가 짧게 끊겼다 다시 잡혀도 NOTIF_REARM_S 안이면 재알람 X.
            # ── 0) 작은 ghost cluster 필터 — 모든 후속 로직은 filtered_tracks 사용
            filtered_tracks = _apply_size_filter(fd.tracks)

            # 두 번째 사람 (다른 id) 이 들어오면 별도 알람 → 클라이언트 corner 카드.
            # filtered_tracks 기준 — 작은 박스는 human_prob=0 으로 demote 되어 알람 X.
            best = 0.0
            count = 0
            for t in filtered_tracks:
                hp = float(t.get("human_prob", 0.0))
                if hp >= HUMAN_THRESHOLD:
                    count += 1
                    if hp > best:
                        best = hp
            present = count > 0
            new_event = False
            for t in filtered_tracks:
                hp = float(t.get("human_prob", 0.0))
                if hp < HUMAN_THRESHOLD:
                    continue
                tid = t.get("id")
                if tid is None:
                    continue
                if tid not in _alarm["active_ids"]:
                    # 새 트랙 등장 → 알람 발생.
                    _alarm["events"] += 1
                    new_event = True
                _alarm["active_ids"][tid] = now
            if present:
                _alarm["last_seen"] = now
            # debounce 시간 지난 트랙은 active_ids 에서 제거 → 다음 등장 시 또 알람.
            expired = [tid for tid, ts in _alarm["active_ids"].items()
                       if now - ts > NOTIF_REARM_S]
            for tid in expired:
                del _alarm["active_ids"][tid]
            # was_present 호환 유지 (구 클라이언트 / 다른 곳 참조 대비).
            _alarm["was_present"] = bool(_alarm["active_ids"])

            # 점 수 cap — Plotly 처럼 figure 통째 보내지 않고
            # 그래도 클라이언트 버퍼 한계 고려해 상한 둠.
            pts = fd.points
            if len(pts) > MAX_POINTS:
                step = len(pts) // MAX_POINTS + 1
                pts = pts[::step]

            # frame number 가 바뀌었을 때만 push (idle 시 트래픽 절약).
            if fd.frame != last_frame_sent:
                # 매칭 JPG basename — 서버에서 bisect 로 계산해 보냄
                # (클라이언트 매번 검색하는 것보다 가벼움, _HAS_JPG=False 면 빈 문자열).
                jpg = find_jpg_basename(fd.timestamp) if _HAS_JPG else None
                msg = {
                    "f": fd.frame,
                    "T": fd.timestamp,   # JPG 매칭용 timestamp (없으면 0)
                    "jpg": jpg or "",
                    "pts": pts,
                    "tracks": filtered_tracks,   # ← 작은 박스는 human_prob=0 으로 demoted
                    "person": {
                        "present": present, "best": best,
                        "count": count, "new": new_event,
                    },
                    "alarm": {
                        "events": _alarm["events"],
                        "last": _alarm["last_seen"],
                    },
                    "age": (now - fd.received_at) if fd.received_at > 0 else -1,
                }
                ws.send(json.dumps(msg))
                last_frame_sent = fd.frame

            time.sleep(REFRESH_MS / 1000.0)
    except Exception as e:
        print(f"[ws] client disconnected: {e}")


# ============ 임베드된 HTML/JS ============
# 전부 한 줄짜리로 응답 — Three.js 는 CDN 에서 import.

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Retina-4SN Viewer (Three.js)</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; overflow: hidden; font-family: -apple-system, BlinkMacSystemFont,
         system-ui, "Segoe UI", sans-serif; background: #f7f7fa; color: #111; }
  #scene { width: 100vw; height: 100vh; display: block; background: #ffffff; }

  /* Standby overlay (사람 없을 때 가시) */
  #standby {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(160deg, #f4f4f7 0%, #e9eaf0 100%);
    display: flex; align-items: center; justify-content: center;
    z-index: 800; transition: opacity 0.3s ease;
  }
  #standby.hidden { opacity: 0; pointer-events: none; }
  .card {
    text-align: center; padding: 32px 48px;
    background: rgba(255,255,255,0.7); backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 28px; box-shadow: 0 12px 48px rgba(0,0,0,0.10);
    border: 0.5px solid rgba(255,255,255,0.6);
  }
  .icon {
    width: 112px; height: 112px; margin: 0 auto 20px;
    background: #e64034; border-radius: 26px;
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 60px;
    animation: pulse 2.4s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(230,64,52,0.45); }
    50%      { transform: scale(1.05); box-shadow: 0 0 0 28px rgba(230,64,52,0); }
  }
  .title { font-size: 30px; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 4px; }
  .clock { font-size: 15px; color: #666; margin-bottom: 24px; }
  .heartbeat {
    display: inline-block; color: #1a7f1a; margin-right: 6px;
    animation: hb 1.6s ease-in-out infinite;
  }
  @keyframes hb {
    0%, 100% { opacity: 0.35; transform: scale(1); }
    50%      { opacity: 1;    transform: scale(1.3); }
  }
  .stats {
    background: rgba(0,0,0,0.025);
    border-radius: 14px; padding: 12px 22px; margin-bottom: 18px;
    text-align: left; min-width: 260px; display: inline-block;
  }
  .stats .row { line-height: 1.95; font-size: 15px; }
  .stats .label { color: #666; width: 92px; display: inline-block; }
  .stats .value { color: #111; font-weight: 700; }
  .hint { font-size: 13px; color: #888; }
  .debug-btn {
    margin-top: 18px; padding: 10px 22px;
    background: rgba(255,255,255,0.78);
    border: 1px solid rgba(0,0,0,0.08); border-radius: 22px;
    color: #555; cursor: pointer; font-family: inherit;
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    font-size: 13px; font-weight: 600;
  }
  .debug-btn:hover { background: rgba(255,255,255,0.95); }

  /* 레이더 모듈 위치 라벨 (origin 마커) */
  .radar-label {
    background: rgba(28, 31, 42, 0.94);
    color: #ffffff;
    padding: 4px 11px;
    border-radius: 7px;
    font-family: ui-monospace, "SF Mono", monospace;
    font-size: 12px; font-weight: 700; letter-spacing: 0.08em;
    white-space: nowrap;
    pointer-events: none;
    transform: translate(-50%, -50%);
    box-shadow: 0 3px 10px rgba(0,0,0,0.35), 0 0 0 1px rgba(255,255,255,0.08);
    text-shadow: 0 0 6px rgba(0,0,0,0.4);
    border: 0.5px solid rgba(230, 64, 52, 0.6);
  }

  /* 3D scene 박스 라벨 (CSS2DRenderer 가 위치 자동 정렬) */
  .box-label {
    font-family: ui-monospace, "SF Mono", Monaco, "Cascadia Code",
                 "Roboto Mono", monospace;
    font-size: 13px; font-weight: 700;
    padding: 3px 9px; border-radius: 7px;
    background: rgba(255,255,255,0.9);
    backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
    box-shadow: 0 2px 8px rgba(0,0,0,0.18);
    border: 0.5px solid rgba(0,0,0,0.08);
    white-space: nowrap;
    pointer-events: none;
    transform: translate(-50%, -100%);   /* 박스 위 중앙 정렬 */
    letter-spacing: -0.01em;
  }

  /* Notification — 두 모드 (center=대기→활성 첫 알림 / corner=활성 중 추가) */
  #notifs {
    position: fixed; z-index: 2000;
    display: flex; flex-direction: column; gap: 16px;
    pointer-events: none;
  }
  #notifs.center {
    left: 50%; top: 50%; transform: translate(-50%, -50%);
    align-items: center;
  }
  #notifs.corner {
    right: 28px; top: 28px;
    align-items: flex-end;
  }

  .notif {
    display: flex; flex-direction: column;
    background: rgba(250,250,252,0.90);
    backdrop-filter: blur(36px) saturate(180%);
    -webkit-backdrop-filter: blur(36px) saturate(180%);
    border: 0.5px solid rgba(255,255,255,0.5);
    transition: opacity 0.35s ease, transform 0.35s ease;
    pointer-events: auto;
  }
  .notif.closing { opacity: 0; pointer-events: none; }

  /* ── Center 모드 (대형, alert dialog 스타일) ── */
  #notifs.center .notif {
    width: 880px; padding: 56px 64px;
    border-radius: 38px; gap: 32px;
    box-shadow: 0 36px 108px rgba(0,0,0,0.38);
  }
  #notifs.center .notif.closing { transform: scale(0.92); }

  /* ── Corner 모드 (작게, 코너 알림) ── */
  #notifs.corner .notif {
    width: 480px; padding: 22px 26px;
    border-radius: 22px; gap: 14px;
    box-shadow: 0 14px 44px rgba(0,0,0,0.25);
  }
  #notifs.corner .notif.closing { transform: translateY(-8px); }

  /* 상단 row: icon + title/time/close + msg */
  .notif-top { display: flex; align-items: center; }
  #notifs.center .notif-top { gap: 28px; }
  #notifs.corner .notif-top { gap: 16px; }

  .notif-icon {
    background: #e64034; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    color: white;
  }
  #notifs.center .notif-icon {
    width: 124px; height: 124px; border-radius: 28px; font-size: 72px;
  }
  #notifs.corner .notif-icon {
    width: 56px; height: 56px; border-radius: 13px; font-size: 32px;
  }

  .notif-body { flex: 1; min-width: 0; }
  .notif-head {
    display: flex; justify-content: space-between;
    align-items: center; gap: 12px;
  }
  .notif-title { font-weight: 700; color: #111; letter-spacing: -0.02em; }
  #notifs.center .notif-title { font-size: 38px; }
  #notifs.corner .notif-title { font-size: 19px; }

  .notif-time-row { display: flex; align-items: center; gap: 10px; flex-shrink: 0; }
  .notif-time { color: #666; }
  #notifs.center .notif-time { font-size: 18px; }
  #notifs.corner .notif-time { font-size: 13px; }

  .notif-close {
    border: none; background: rgba(0,0,0,0.07);
    border-radius: 50%; padding: 0; color: #444;
    cursor: pointer; font-family: inherit; flex-shrink: 0;
  }
  .notif-close:hover { background: rgba(0,0,0,0.15); }
  #notifs.center .notif-close {
    width: 42px; height: 42px; line-height: 40px; font-size: 20px;
  }
  #notifs.corner .notif-close {
    width: 26px; height: 26px; line-height: 24px; font-size: 13px;
  }

  .notif-msg { color: #333; display: block; line-height: 1.4; }
  #notifs.center .notif-msg { font-size: 26px; margin-top: 10px; }
  #notifs.corner .notif-msg { font-size: 17px; margin-top: 4px; }

  /* 확인 버튼 */
  .notif-actions {
    display: flex; justify-content: flex-end; gap: 12px;
  }
  .notif-confirm {
    background: #e64034; color: white;
    border: none; cursor: pointer; font-family: inherit;
    font-weight: 700; letter-spacing: -0.01em;
    transition: background 0.18s ease, transform 0.08s ease, box-shadow 0.18s ease;
    box-shadow: 0 6px 18px rgba(230,64,52,0.32);
  }
  .notif-confirm:hover {
    background: #d12d23; box-shadow: 0 8px 22px rgba(230,64,52,0.45);
  }
  .notif-confirm:active { transform: scale(0.97); }
  #notifs.center .notif-confirm {
    padding: 18px 56px; font-size: 22px; border-radius: 18px;
    min-width: 180px;
  }
  #notifs.corner .notif-confirm {
    padding: 8px 22px; font-size: 14px; border-radius: 11px;
  }

  /* "오탐 보고" 버튼 — confirm 옆 회색 보조 버튼 */
  .notif-report {
    background: rgba(0,0,0,0.06); color: #444;
    border: none; cursor: pointer; font-family: inherit;
    font-weight: 600; letter-spacing: -0.01em;
    transition: background 0.18s ease, transform 0.08s ease;
  }
  .notif-report:hover { background: rgba(0,0,0,0.12); }
  .notif-report:active { transform: scale(0.97); }
  #notifs.center .notif-report {
    padding: 18px 36px; font-size: 18px; border-radius: 18px;
  }
  #notifs.corner .notif-report {
    padding: 8px 16px; font-size: 13px; border-radius: 11px;
  }

  /* 오탐 보고 modal */
  #reportModal {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.55);
    display: none; align-items: center; justify-content: center;
    z-index: 3000;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
  }
  #reportModal.show { display: flex; }
  .report-card {
    background: #ffffff;
    border-radius: 24px;
    padding: 36px 44px;
    width: 520px; max-width: 90vw;
    box-shadow: 0 24px 80px rgba(0,0,0,0.40);
    font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
  }
  .report-title {
    font-size: 24px; font-weight: 700; color: #111;
    letter-spacing: -0.02em; margin-bottom: 6px;
  }
  .report-sub {
    font-size: 14px; color: #666; margin-bottom: 22px;
  }
  .report-label-row {
    font-size: 13px; color: #555; font-weight: 600;
    margin-bottom: 8px;
  }
  .report-buttons {
    display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
    margin-bottom: 18px;
  }
  .report-choice {
    padding: 14px; border-radius: 12px;
    border: 1.5px solid rgba(0,0,0,0.08);
    background: rgba(0,0,0,0.02);
    cursor: pointer; font-family: inherit;
    font-size: 15px; font-weight: 600; color: #333;
    transition: all 0.15s ease;
  }
  .report-choice:hover {
    border-color: #e64034;
    background: rgba(230,64,52,0.04);
  }
  .report-choice.selected {
    border-color: #e64034;
    background: rgba(230,64,52,0.1);
    color: #c52a2a;
  }
  .report-note {
    width: 100%; min-height: 70px; padding: 12px;
    border-radius: 10px;
    border: 1.5px solid rgba(0,0,0,0.08);
    font-family: inherit; font-size: 14px; color: #222;
    box-sizing: border-box;
    resize: vertical;
    margin-bottom: 18px;
  }
  .report-note:focus {
    outline: none; border-color: #e64034;
  }
  .report-actions {
    display: flex; justify-content: flex-end; gap: 10px;
  }
  .report-cancel {
    padding: 11px 22px; border-radius: 12px;
    border: 1.5px solid rgba(0,0,0,0.08);
    background: white; color: #555; cursor: pointer;
    font-family: inherit; font-size: 14px; font-weight: 600;
  }
  .report-cancel:hover { background: rgba(0,0,0,0.04); }
  .report-submit {
    padding: 11px 28px; border-radius: 12px;
    border: none; background: #e64034; color: white;
    cursor: pointer; font-family: inherit;
    font-size: 14px; font-weight: 700;
    box-shadow: 0 4px 14px rgba(230,64,52,0.32);
  }
  .report-submit:hover { background: #d12d23; }
  .report-submit:disabled {
    background: #ccc; color: #888; cursor: not-allowed;
    box-shadow: none;
  }
  .report-status {
    font-size: 13px; color: #888; margin-top: 12px; text-align: center;
  }

  /* Dashboard stats (좌하단) */
  #dashStats {
    position: fixed; left: 16px; bottom: 52px; z-index: 1000;
    font-size: 15px; color: #222;
    background: rgba(255,255,255,0.9);
    padding: 12px 16px; border-radius: 12px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    pointer-events: none; min-width: 200px;
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
  }
  #dashStats .row { line-height: 1.7; }
  #dashStats .label { color: #777; width: 92px; display: inline-block; }
  #dashStats .value { color: #111; font-weight: 700; }
  #dashStats .debug-on { color: #c52a2a; font-weight: 700; font-size: 13px;
                         padding-bottom: 6px; border-bottom: 1px solid rgba(0,0,0,0.08);
                         margin-bottom: 6px; }

  /* Legend (좌하단, dashStats 위) */
  #legend {
    position: fixed; left: 16px; bottom: 178px; z-index: 1000;
    font-size: 14px; color: #222;
    background: rgba(255,255,255,0.9);
    padding: 7px 14px; border-radius: 10px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    pointer-events: none;
  }
  #legend .item { margin-right: 14px; }
  #legend .dot { margin-right: 5px; }

  /* Control bar (하단 중앙) */
  #controls {
    position: fixed; left: 50%; bottom: 26px; z-index: 1500;
    transform: translateX(-50%);
    display: flex; gap: 18px;
  }
  .ctrl {
    display: flex; flex-direction: column; align-items: center;
    gap: 3px; min-width: 104px;
    padding: 12px 22px; cursor: pointer;
    border: 1px solid rgba(0,0,0,0.08); border-radius: 18px;
    background: rgba(255,255,255,0.9);
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    box-shadow: 0 4px 16px rgba(0,0,0,0.14);
    font-family: inherit;
  }
  .ctrl:hover { background: rgba(255,255,255,1.0); }
  .ctrl-icon { font-size: 30px; line-height: 1; }
  .ctrl-label { font-size: 13px; color: #333; font-weight: 600; }

  /* Connection indicator (좌상단) */
  #connStatus {
    position: fixed; top: 12px; left: 12px; z-index: 1000;
    font-size: 12px; padding: 4px 10px;
    background: rgba(255,255,255,0.85);
    border-radius: 6px; font-family: ui-monospace, monospace;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    transition: color 0.3s ease;
  }

  .hidden-el { display: none !important; }

  /* ── 좌측 JPG 패널 (data-has-jpg="true" 일 때만 활성) ──────────── */
  body[data-has-jpg="true"] #jpgPanel {
    position: fixed; top: 0; left: 0;
    width: 50vw; height: 100vh;
    background: #000; z-index: 100; overflow: hidden;
  }
  body[data-has-jpg="false"] #jpgPanel { display: none; }
  #jpgPanel img {
    width: 100%; height: 100%; object-fit: contain; background: #000;
  }
  #jpgLabel {
    position: absolute; top: 16px; left: 16px;
    padding: 6px 14px; background: rgba(0,0,0,0.6); color: #fff;
    border-radius: 8px; font-size: 13px; font-weight: 600; z-index: 10;
  }
  #jpgStatus {
    position: absolute; bottom: 16px; left: 16px;
    padding: 5px 10px; background: rgba(0,0,0,0.6); color: #fff;
    border-radius: 8px; font-family: ui-monospace, monospace;
    font-size: 11px; z-index: 10;
  }
  /* JPG 있을 때 우측 viewer 영역으로 모든 fixed UI 이동 */
  body[data-has-jpg="true"] #scene { left: 50vw !important; width: 50vw !important; }
  body[data-has-jpg="true"] #standby { left: 50vw !important; }
  body[data-has-jpg="true"] #legend { left: calc(50vw + 16px) !important; }
  body[data-has-jpg="true"] #dashStats { left: calc(50vw + 16px) !important; }
  body[data-has-jpg="true"] #controls { left: 75vw !important; }
  body[data-has-jpg="true"] #connStatus { left: calc(50vw + 12px) !important; }
</style>
</head>
<body data-has-jpg="__HAS_JPG__">

<!-- 좌측 JPG 패널 (--jpg-dir 옵션 시) -->
<div id="jpgPanel">
  <img id="jpgFrame" src="" alt="">
  <div id="jpgLabel">📷 카메라 (참고)</div>
  <div id="jpgStatus">대기 중</div>
</div>

<canvas id="scene" style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;"></canvas>

<div id="connStatus">연결 중...</div>

<!-- Standby overlay -->
<div id="standby">
  <div class="card">
    <div class="icon">&#x1F464;</div>
    <div class="title">감시 중</div>
    <div class="clock">
      <span class="heartbeat">●</span>
      <span id="clockTime">--:--:--</span>
    </div>
    <div class="stats">
      <div class="row"><span class="label">누적 감지</span><span class="value" id="sbEvents">0회</span></div>
      <div class="row"><span class="label">마지막</span><span class="value" id="sbLast">기록 없음</span></div>
      <div class="row"><span class="label">연결</span><span class="value" id="sbSignal">대기 중</span></div>
      <div class="row"><span class="label">화면 속도</span><span class="value" id="sbFps">0/초</span></div>
    </div>
    <div class="hint">사람이 감지되면 자동으로 화면이 전환됩니다</div>
    <br>
    <button class="debug-btn" id="debugBtn">🔧 대시보드 강제 열기 (디버그)</button>
  </div>
</div>

<!-- Dashboard UI (사람 검출 시) -->
<div id="legend" class="hidden-el">
  <span class="item"><span class="dot" style="color:#1da4e6">●</span>레이더 점</span>
</div>

<div id="dashStats" class="hidden-el">
  <div class="debug-on hidden-el" id="dsDebugBadge">🔧 디버그 모드 — '대기로' 클릭 시 종료</div>
  <div class="row"><span class="label">현재 사람</span><span class="value" id="dsCount">0명</span></div>
  <div class="row"><span class="label">총 감지</span><span class="value" id="dsEvents">0회</span></div>
  <div class="row"><span class="label">레이더 점</span><span class="value" id="dsPts">0개</span></div>
  <div class="row"><span class="label">화면 속도</span><span class="value" id="dsFps">0/초</span></div>
  <div class="row"><span class="label">신호 상태</span><span class="value" id="dsSignal">--</span></div>
</div>

<div id="controls" class="hidden-el">
  <button class="ctrl" id="resetBtn">
    <span class="ctrl-icon">⟳</span>
    <span class="ctrl-label">시점 리셋</span>
  </button>
  <button class="ctrl" id="standbyBtn">
    <span class="ctrl-icon">🛏</span>
    <span class="ctrl-label">대기로</span>
  </button>
</div>

<div id="notifs" class="center"></div>

<!-- 오탐 보고 modal -->
<div id="reportModal">
  <div class="report-card">
    <div class="report-title">오탐 보고</div>
    <div class="report-sub">사람이 감지됐다고 떴는데 실제로는 다른 것이었다면 알려주세요.<br>최근 30초 데이터가 학습용으로 저장됩니다.</div>
    <div class="report-label-row">무엇이었나요?</div>
    <div class="report-buttons" id="reportChoices">
      <button class="report-choice" data-label="noise">노이즈 / 잘못된 클러스터</button>
      <button class="report-choice" data-label="animal">동물</button>
      <button class="report-choice" data-label="object">정적 객체 (가구 등)</button>
      <button class="report-choice" data-label="other">기타</button>
    </div>
    <div class="report-label-row">추가 메모 (선택)</div>
    <textarea id="reportNote" class="report-note" placeholder="예: 의자에 옷이 걸려 있었음"></textarea>
    <div class="report-actions">
      <button class="report-cancel" id="reportCancel">취소</button>
      <button class="report-submit" id="reportSubmit" disabled>보고하기</button>
    </div>
    <div class="report-status" id="reportStatus"></div>
  </div>
</div>

<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.159.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.159.0/examples/jsm/"
  }
}
</script>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
// 굵은 line 용 (WebGL 의 LineBasicMaterial.linewidth=1 한계 우회).
// LineSegments2 는 shader 로 fat line 그려서 pixel 단위 두께 지원.
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
// 3D 좌표 → 화면 좌표 자동 정렬되는 HTML 라벨 오버레이.
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

// ╔══════════════════════════════════════════════════════════════╗
// ║  시각화 설정 — 여기 값만 바꾸면 색상/임계값 즉시 반영           ║
// ║  RGB 는 0~1 정규화. 예: 빨강 = [1.0, 0.2, 0.2]                ║
// ╚══════════════════════════════════════════════════════════════╝
// 사람으로 분류된 cluster 의 점/엣지 색.
const PERSON_COLOR  = [0.10, 0.82, 0.30];   // 기본: 초록  /  빨강: [0.95, 0.20, 0.20]
// 추론 대기 중 cluster (track 형성됐지만 cascade 추론 아직 안 끝남, human_prob<0).
// 트랙 = 중요 → 잘 보이는 파랑 (흰 배경 대비 큼).
const PENDING_COLOR = [0.11, 0.64, 0.90];   // deepskyblue
// 일반 cluster (사람 아님 / 추론 끝나고 비사람으로 분류됨) 의 점/엣지 색.
const CLUSTER_COLOR = [0.30, 0.70, 0.95];   // 밝은 파랑 (pending 보다 조금 더 밝게)
// 노이즈 점 (DBSCAN -1 = 어느 cluster 에도 속하지 않음).
// 시각 우선순위 낮음 → 흰 배경에 안 두드러지는 노랑.
const NOISE_COLOR   = [1.00, 0.85, 0.10];   // 노랑
// 사람으로 한 번 분류된 track id 가 이 시간(ms)동안 "사람 색" 유지.
// 분류값 잠시 떨어지거나 트래킹 끊겨도 점 색 깜빡임 방지.
const PERSON_COLOR_PERSIST_MS = 3000;

// ── Z-up 좌표계 (radar/Plotly 와 동일) ──────────────────────────────
THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

// ── Scene / Camera / Renderer ───────────────────────────────────
const canvas = document.getElementById('scene');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);

// canvas 가 _HAS_JPG 에 따라 100vw 또는 50vw 차지 — getBoundingClientRect 기준.
function canvasSize() {
  const r = canvas.getBoundingClientRect();
  return { w: r.width || window.innerWidth, h: r.height || window.innerHeight };
}
const _initSize = canvasSize();
const camera = new THREE.PerspectiveCamera(
  50, _initSize.w / _initSize.h, 0.1, 100);
camera.up.set(0, 0, 1);
const INITIAL_CAM = { pos: [3.5, -4.5, 3.0], target: [0, 2.5, 0] };
camera.position.set(...INITIAL_CAM.pos);

const renderer = new THREE.WebGLRenderer({ antialias: true, canvas });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(_initSize.w, _initSize.h, false);

// CSS2D 라벨 오버레이 — HTML element 를 3D 좌표에 자동 정렬.
// WebGL canvas 위에 절대 위치로 깔리며, pointer-events: none 으로 클릭 통과.
const labelRenderer = new CSS2DRenderer();
labelRenderer.setSize(_initSize.w, _initSize.h);
labelRenderer.domElement.style.position = 'absolute';
labelRenderer.domElement.style.top = '0';
labelRenderer.domElement.style.left = '0';
labelRenderer.domElement.style.pointerEvents = 'none';
labelRenderer.domElement.style.zIndex = '100';   // canvas 위, UI 패널 아래
document.body.appendChild(labelRenderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(...INITIAL_CAM.target);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.update();

// ── Grid + 축 (참조용) ──────────────────────────────────────────
const grid = new THREE.GridHelper(6, 12, 0xbbbbbb, 0xdddddd);
grid.rotation.x = Math.PI / 2;      // XY 평면에 눕히기 (z-up)
grid.position.set(0, 3, -1.5);
scene.add(grid);

// 바닥 코너 reference (얇게, 보조)
const axes = new THREE.AxesHelper(0.5);
axes.position.set(0, 0, -1.5);
scene.add(axes);

// ── 레이더 모듈 위치 마커 (origin) ─────────────────────────────
// 사용자 시인성 강화 — 실제 device 형상 (어두운 박스) + 두꺼운 RGB axes + 라벨.
// 점 cloud 가 화이트 배경에 노이즈처럼 흩어져 있어서 원점이 안 보였던 문제 해결.
const radarBox = new THREE.Mesh(
  new THREE.BoxGeometry(0.22, 0.08, 0.10),     // 22cm × 8cm × 10cm (벽 마운트 모듈)
  new THREE.MeshBasicMaterial({color: 0x1c1f2a})
);
radarBox.position.set(0, -0.04, 0);             // y 살짝 뒤로 (벽에 붙은 느낌)
scene.add(radarBox);

// 빨간 액센트 (전면 LED 같은 점) — 어디가 정면(+Y 방향)인지 표시
const radarFront = new THREE.Mesh(
  new THREE.SphereGeometry(0.025, 12, 8),
  new THREE.MeshBasicMaterial({color: 0xe64034})
);
radarFront.position.set(0, 0.01, 0);
scene.add(radarFront);

// 두꺼운 axes — origin 에서 0.5m 뻗음. LineSegments2 fat line.
const originAxesGeo = new LineSegmentsGeometry();
const ORIGIN_AXIS_LEN = 0.5;
originAxesGeo.setPositions([
  0,0,0,  ORIGIN_AXIS_LEN,0,0,    // +X
  0,0,0,  0,ORIGIN_AXIS_LEN,0,    // +Y (정면)
  0,0,0,  0,0,ORIGIN_AXIS_LEN,    // +Z (위)
]);
originAxesGeo.setColors([
  0.95,0.20,0.20,  0.95,0.20,0.20,   // X red
  0.20,0.85,0.30,  0.20,0.85,0.30,   // Y green
  0.20,0.55,0.95,  0.20,0.55,0.95,   // Z blue
]);
const originAxesMat = new LineMaterial({
  vertexColors: true,
  linewidth: 4,
  worldUnits: false,
  resolution: new THREE.Vector2(window.innerWidth, window.innerHeight),
});
const originAxes = new LineSegments2(originAxesGeo, originAxesMat);
originAxes.frustumCulled = false;
scene.add(originAxes);

// "RADAR" 라벨 (디바이스 위)
const radarLabelEl = document.createElement('div');
radarLabelEl.className = 'radar-label';
radarLabelEl.textContent = 'RADAR';
const radarLabelObj = new CSS2DObject(radarLabelEl);
radarLabelObj.position.set(0, 0, 0.16);    // 디바이스 위 16cm
scene.add(radarLabelObj);

// ── 점 cloud (BufferGeometry, in-place 갱신) ────────────────────
const MAX_POINTS = 5000;
const ptsGeo = new THREE.BufferGeometry();
const ptsPos = new Float32Array(MAX_POINTS * 3);
const ptsCol = new Float32Array(MAX_POINTS * 3);
ptsGeo.setAttribute('position', new THREE.BufferAttribute(ptsPos, 3));
ptsGeo.setAttribute('color',    new THREE.BufferAttribute(ptsCol, 3));
ptsGeo.setDrawRange(0, 0);
const ptsMat = new THREE.PointsMaterial({
  vertexColors: true, size: 0.08, sizeAttenuation: true,
});
const pointsObj = new THREE.Points(ptsGeo, ptsMat);
scene.add(pointsObj);

// ── 클러스터 내 점-점 연결선 (객체 윤곽 강조) ─────────────────────
// 같은 track_id (>= 0) 인 점들 중, 일정 거리 (CLUSTER_EDGE_MAX_DIST m) 이내
// 페어를 LineSegments 로 연결 → 사람 윤곽이 web/mesh 형태로 드러남.
// 노이즈 점 (track_id < 0) 은 연결 X.
const CLUSTER_EDGE_MAX_DIST = 0.32;   // m — 너무 크면 사람 영역 외 점들도 이어짐
const MAX_EDGES = 6000;
const edgeGeo = new THREE.BufferGeometry();
const edgePos = new Float32Array(MAX_EDGES * 2 * 3);
const edgeCol = new Float32Array(MAX_EDGES * 2 * 3);
edgeGeo.setAttribute('position', new THREE.BufferAttribute(edgePos, 3));
edgeGeo.setAttribute('color',    new THREE.BufferAttribute(edgeCol, 3));
edgeGeo.setDrawRange(0, 0);
const edgeMat = new THREE.LineBasicMaterial({
  vertexColors: true,
  transparent: true, opacity: 0.45,   // 반투명 — 점이 묻히지 않게
  depthWrite: false,                  // 배경 점들 가려지지 않게
});
const edgeLines = new THREE.LineSegments(edgeGeo, edgeMat);
scene.add(edgeLines);

// ── 박스 (LineSegments2 — fat line, pixel 두께 지원) ─────────────
// WebGL 의 LineBasicMaterial 은 linewidth=1 고정 (대부분 GPU 드라이버 한계).
// LineSegments2 는 fragment shader 로 두꺼운 line 그려서 pixel 단위 두께 OK.
const BOX_LINE_WIDTH = 5;   // pixels
const boxGeo = new LineSegmentsGeometry();
const boxMat = new LineMaterial({
  vertexColors: true,
  linewidth: BOX_LINE_WIDTH,
  worldUnits: false,     // pixel 기준 (true 면 world 단위 — 거리에 따라 크기 변화)
  resolution: new THREE.Vector2(window.innerWidth, window.innerHeight),
});
const boxLines = new LineSegments2(boxGeo, boxMat);
boxLines.frustumCulled = false;   // bbox 미정의 시 사라지지 않게.
scene.add(boxLines);

// 0/1 → min/max 패턴으로 박스 12 edges 표현.
const BOX_EDGES = [
  // bottom z=0
  [0,0,0, 1,0,0], [1,0,0, 1,1,0], [1,1,0, 0,1,0], [0,1,0, 0,0,0],
  // top z=1
  [0,0,1, 1,0,1], [1,0,1, 1,1,1], [1,1,1, 0,1,1], [0,1,1, 0,0,1],
  // vertical
  [0,0,0, 0,0,1], [1,0,0, 1,0,1], [1,1,0, 1,1,1], [0,1,0, 0,1,1],
];

// 박스 라벨 풀 — CSS2DObject 들을 재사용. setBoxes 시 필요한 만큼 visible 토글.
const labelPool = [];   // [{obj: CSS2DObject, el: HTMLDivElement}, ...]
function getLabel(idx) {
  if (labelPool[idx]) return labelPool[idx];
  const el = document.createElement('div');
  el.className = 'box-label';
  const obj = new CSS2DObject(el);
  scene.add(obj);
  const slot = { obj, el };
  labelPool[idx] = slot;
  return slot;
}

function setBoxes(boxes) {
  // 박스 라인 그리기 비활성 — 사람 영역은 점/엣지 초록 색으로 표시.
  // 라벨만 박스 위치에 띄움 (추론 % 값 보이게).
  boxLines.visible = false;

  // 라벨 — 풀 재사용. 각 박스 위 (top z + 0.18) 에 표시.
  // 라벨 텍스트 비어 있으면 (추론 대기 노란 박스) 숨김.
  for (let i = 0; i < boxes.length; i++) {
    const b = boxes[i];
    const slot = getLabel(i);
    if (!b.label) {
      slot.obj.visible = false;
      continue;
    }
    slot.obj.visible = true;
    slot.el.textContent = b.label;
    // 박스 상단 중앙 위에 띄움.
    const cx = (b.bbox[0] + b.bbox[1]) / 2;
    const cy = (b.bbox[2] + b.bbox[3]) / 2;
    const cz = b.bbox[5] + 0.18;
    slot.obj.position.set(cx, cy, cz);
    // 라벨 글자색을 박스 색에 맞춤 (가독성 위해 어둡게 조정).
    const r = Math.round(b.color[0] * 200);
    const g = Math.round(b.color[1] * 200);
    const bl = Math.round(b.color[2] * 200);
    slot.el.style.color = `rgb(${r},${g},${bl})`;
  }
  // 남은 풀 슬롯은 모두 숨김 (트랙 개수가 줄어든 경우).
  for (let i = boxes.length; i < labelPool.length; i++) {
    labelPool[i].obj.visible = false;
  }
}

function updatePoints(rawPts, personIds, pendingIds) {
  const n = Math.min(rawPts.length, MAX_POINTS);
  // 우선순위: pending(노랑, 추론 대기) > person(초록) > cluster(파랑) > 노이즈.
  const [PR, PG, PB] = PERSON_COLOR;
  const [YR, YG, YB] = PENDING_COLOR;
  const [CR, CG, CB] = CLUSTER_COLOR;
  const [NR, NG, NB] = NOISE_COLOR;
  for (let i = 0; i < n; i++) {
    const p = rawPts[i];
    ptsPos[i*3+0] = p[0];
    ptsPos[i*3+1] = p[1];
    ptsPos[i*3+2] = p[2];
    const tid = p[5] | 0;
    if (tid >= 0 && pendingIds && pendingIds.has(tid)) {
      ptsCol[i*3+0] = YR; ptsCol[i*3+1] = YG; ptsCol[i*3+2] = YB;
    } else if (tid >= 0 && personIds && personIds.has(tid)) {
      ptsCol[i*3+0] = PR; ptsCol[i*3+1] = PG; ptsCol[i*3+2] = PB;
    } else if (tid >= 0) {
      ptsCol[i*3+0] = CR; ptsCol[i*3+1] = CG; ptsCol[i*3+2] = CB;
    } else {
      ptsCol[i*3+0] = NR; ptsCol[i*3+1] = NG; ptsCol[i*3+2] = NB;
    }
  }
  ptsGeo.setDrawRange(0, n);
  ptsGeo.attributes.position.needsUpdate = true;
  ptsGeo.attributes.color.needsUpdate = true;
}

// 같은 cluster (track_id >= 0) 인 점 쌍 중 거리 < CLUSTER_EDGE_MAX_DIST 인 것
// 만 연결. O(N²) 이지만 cluster 당 점 수 보통 < 100 이라 비용 미미.
// 사람 cluster 의 edge 는 초록, 나머지는 파랑.
function buildClusterEdges(rawPts, personIds, pendingIds) {
  // track_id 별로 cluster 의 점 인덱스 그룹화.
  const clusters = new Map();
  for (let i = 0; i < rawPts.length; i++) {
    const tid = rawPts[i][5];
    if (tid < 0) continue;   // 노이즈 점은 제외
    if (!clusters.has(tid)) clusters.set(tid, []);
    clusters.get(tid).push(rawPts[i]);
  }

  const MAX_DIST_SQ = CLUSTER_EDGE_MAX_DIST * CLUSTER_EDGE_MAX_DIST;
  // edge 색은 점 색 그대로 사용 (상단 상수). 우선순위 동일.
  const [PR, PG, PB] = PERSON_COLOR;
  const [YR, YG, YB] = PENDING_COLOR;
  const [CR, CG, CB] = CLUSTER_COLOR;
  let v = 0;

  for (const [tid, pts] of clusters) {
    const isPending = pendingIds && pendingIds.has(tid);
    const isPerson = personIds && personIds.has(tid);
    let R, G, B;
    if (isPending)      { R = YR; G = YG; B = YB; }
    else if (isPerson)  { R = PR; G = PG; B = PB; }
    else                { R = CR; G = CG; B = CB; }
    const cn = pts.length;
    for (let i = 0; i < cn - 1; i++) {
      const pi = pts[i];
      for (let j = i + 1; j < cn; j++) {
        const pj = pts[j];
        const dx = pi[0] - pj[0];
        const dy = pi[1] - pj[1];
        const dz = pi[2] - pj[2];
        const dsq = dx*dx + dy*dy + dz*dz;
        if (dsq > MAX_DIST_SQ) continue;
        if (v >= MAX_EDGES) break;
        const off = v * 6;
        edgePos[off+0] = pi[0]; edgePos[off+1] = pi[1]; edgePos[off+2] = pi[2];
        edgePos[off+3] = pj[0]; edgePos[off+4] = pj[1]; edgePos[off+5] = pj[2];
        edgeCol[off+0] = R; edgeCol[off+1] = G; edgeCol[off+2] = B;
        edgeCol[off+3] = R; edgeCol[off+4] = G; edgeCol[off+5] = B;
        v++;
      }
      if (v >= MAX_EDGES) break;
    }
    if (v >= MAX_EDGES) break;
  }

  edgeGeo.setDrawRange(0, v * 2);
  edgeGeo.attributes.position.needsUpdate = true;
  edgeGeo.attributes.color.needsUpdate = true;
}

// ── 트랙 → trackState 갱신 (네트워크 frame 수신 시) ─────────────
// 박스 위치는 즉시 그리지 않고 target 만 저장. animate() 가 lerp 로 그림.
//
// 공간 우선 매칭 — id 가 충돌하든 말든 무관:
//   1) 매 frame, 각 track 을 trackState 의 기존 entry 중 공간적으로 가장 가까운 것과 매칭
//      (한 entry 는 한 번만 consumed — 한 box 가 두 track 을 동시에 받는 일 없음)
//   2) 매칭 안 된 track 은 새 unique key 로 추가 → 모든 track 이 보장되게 entry 받음
// → ID 가 3개 다 같든, 3개 unique 든, 일부 같든 상관없이 항상 3 entry 보장.
const SPATIAL_MATCH_M = 1.5;  // 동일 트랙 식별 거리 (m, xy centroid 기준)

// ── 사람 분류 track id 색상 cache ──────────────────────────────
// 한 번 사람으로 분류된 track id 는 PERSON_COLOR_PERSIST_MS 동안 "사람 색" 유지.
// inference 값이 잠시 떨어지거나 track 이 잠시 끊겨도 점 색이 안 깜빡임.
// 상수는 파일 상단 시각화 설정 블록에 있음.
const personIdCache = new Map();   // tid → expiresAt (performance.now ms)

function updateTrackTargets(tracks) {
  const now = performance.now();
  lerpAndDrawBoxes._lastRawTracks = tracks.length;

  // ── 단순 1:1 모드 — trackState 를 매 frame 서버 tracks 와 동기화. ──
  // 사용자 요청: "추론 결과 있는 곳에 무조건 박스 쳐라"
  // 매칭 로직 부작용 (신규가 기존 entry 뺏기 등) 자체를 제거.
  //   1) 각 track 에 대해 가능하면 기존 entry 의 current 위치 이어받음 (lerp 연속성).
  //   2) 이번 frame 에 매칭 안 된 OLD entry 는 즉시 삭제 (persistence X).
  //      → 모듈이 1 frame 만 track 빼먹어도 그 frame 박스 사라짐.
  //      → 안 빼먹으면 = 매 frame N tracks 면 항상 N 박스.
  const newKeys = new Set();
  const newState = new Map();

  for (let i = 0; i < tracks.length; i++) {
    const t = tracks[i];
    const bbox = t.bbox || [0,0,0,0,0,0];
    const baseId = t.id !== undefined ? String(t.id) : '?';
    const prob = t.human_prob !== undefined ? t.human_prob : 0;
    const tcx = (bbox[0] + bbox[1]) / 2;
    const tcy = (bbox[2] + bbox[3]) / 2;

    // 기존 entry 중 공간적으로 가장 가까운 것 찾기 (lerp 위치 이어받기 용).
    let bestKey = null;
    let bestDist = SPATIAL_MATCH_M;
    for (const [k, s] of trackState) {
      if (newKeys.has(k)) continue;
      const scx = (s.current[0] + s.current[1]) / 2;
      const scy = (s.current[2] + s.current[3]) / 2;
      const d = Math.hypot(tcx - scx, tcy - scy);
      if (d < bestDist) { bestDist = d; bestKey = k; }
    }

    // 키 결정 — 기존 매칭 있으면 그 키, 없으면 baseId (충돌 시 #suffix).
    let key;
    let current;
    if (bestKey !== null) {
      key = bestKey;
      current = trackState.get(bestKey).current.slice();   // 부드러운 이어받기
    } else {
      key = baseId;
      let suffix = 0;
      while (newKeys.has(key)) { suffix++; key = `${baseId}#${suffix}`; }
      current = bbox.slice();   // 신규 — 바로 target 위치
    }
    newKeys.add(key);

    let color, label;
    if (prob < 0) {
      color = [1.0, 0.92, 0.0];
      label = '';
    } else if (prob >= SEQ_THRESHOLD) {
      color = [0.0, 0.85, 0.0];
      label = `id ${baseId} | human ${(prob*100).toFixed(1)}% (seq)`;
    } else {
      color = [0.5, 0.5, 0.5];
      label = `id ${baseId} | human ${(prob*100).toFixed(1)}% (seq)`;
    }
    newState.set(key, {
      current,
      target: bbox.slice(),
      color, label, lastSeenMs: now,
    });
  }

  // OLD trackState 의 매칭 안 된 entry 전부 제거 + newState 로 교체.
  trackState.clear();
  for (const [k, v] of newState) trackState.set(k, v);
}

// ── 매 render frame 호출 — current → target lerp 후 setBoxes ──
// 모듈 frame 수신 사이에도 60 Hz 로 부드럽게 보간된 박스가 그려진다.
function lerpAndDrawBoxes() {
  const now = performance.now();
  // trackState 는 updateTrackTargets 가 매 frame 갱신 — 별도 timeout 삭제 불필요.
  // current → target 보간 + 박스 배열 구성.
  const boxes = [];
  for (const s of trackState.values()) {
    for (let i = 0; i < 6; i++) {
      s.current[i] += (s.target[i] - s.current[i]) * LERP_FACTOR;
    }
    boxes.push({bbox: s.current, color: s.color, label: s.label});
  }
  // 진단 로그 — 박스 수 변화 시 출력. raw tracks vs trackState vs 그려진 boxes.
  if (boxes.length !== lerpAndDrawBoxes._lastCount) {
    const rawCount = lerpAndDrawBoxes._lastRawTracks;
    console.log(`[viewer] raw_tracks=${rawCount} trackState=${trackState.size} boxes=${boxes.length}`);
    if (boxes.length >= 2 || rawCount >= 2) {
      let i = 0;
      for (const [k, s] of trackState) {
        const cx = (s.current[0] + s.current[1]) / 2;
        const cy = (s.current[2] + s.current[3]) / 2;
        console.log(`  [${i++}] key="${k}" label="${s.label}" centroid=(${cx.toFixed(2)},${cy.toFixed(2)})`);
      }
    }
    lerpAndDrawBoxes._lastCount = boxes.length;
  }
  setBoxes(boxes);
}
lerpAndDrawBoxes._lastCount = -1;
lerpAndDrawBoxes._lastRawTracks = 0;   // updateTrackTargets 가 매 frame 갱신

// ── App state ───────────────────────────────────────────────────
const SEQ_THRESHOLD = 0.7;
// 사람 사라진 후 dashboard 유지 시간 — 짧으면 다시 들어왔을 때 다시 화면 전환되어 산만.
const ACTIVE_HOLD_S = 30;

let activeUntilMs = 0;   // 사람 사라진 후 이 시각까지 active 유지
let manualMode = false;  // 디버그 강제 활성
let lastFrame = -1;
let frameCount = 0;
let frameWindowStart = performance.now();
let currentFps = 0;

// ── 박스 보간 (옵션 A) ─────────────────────────────────────────
// 모듈 처리 rate (5~10 Hz) 와 brower render rate (60 Hz) 의 비동기 차이가
// "뚝뚝 끊김" 체감을 키움. trackState 에 current/target 따로 두고
// 매 render frame 에서 current → target 으로 LERP 하면 시각적으로 부드러움.
// id → {current: [6 floats], target: [6 floats], color, label, lastSeenMs}
const trackState = new Map();
// 매 frame current 가 target 으로 LERP_FACTOR 만큼 이동.
// 0.18: ~16 frame (≈270 ms) 안에 95% 도달. 너무 크면 보간 없음, 너무 작으면 지연 큼.
const LERP_FACTOR = 0.18;
// 1.5초 이상 새 데이터 없으면 트랙 제거 (서버 측 트래킹 종료 응답).
const TRACK_TIMEOUT_MS = 1500;

// ── Animation loop ──────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  // 매 render frame 에서 박스 lerp + setBoxes — 모듈 fps 가 낮아도 60 fps 부드러움.
  lerpAndDrawBoxes();
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);   // 라벨 (CSS2D) 도 매 frame 재정렬.
}
animate();

window.addEventListener('resize', () => {
  const { w, h } = canvasSize();
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
  // LineMaterial 은 pixel 두께 계산에 화면 해상도 필요 — resize 마다 갱신.
  boxMat.resolution.set(w, h);
  originAxesMat.resolution.set(w, h);
  labelRenderer.setSize(w, h);
});

// ── 컨트롤 버튼 ────────────────────────────────────────────────
document.getElementById('resetBtn').addEventListener('click', () => {
  camera.position.set(...INITIAL_CAM.pos);
  controls.target.set(...INITIAL_CAM.target);
  controls.update();
});

document.getElementById('standbyBtn').addEventListener('click', () => {
  manualMode = false;
  activeUntilMs = 0;
  applyMode();
});

document.getElementById('debugBtn').addEventListener('click', () => {
  manualMode = true;
  applyMode();
});

function applyMode() {
  const isActive = manualMode || (Date.now() < activeUntilMs);
  // active → standby: 다음 알림은 다시 center 로.
  if (prevActiveForNotif && !isActive) {
    centerNextNotif = true;
  }
  // standby → active: 이미 dashboard 가 뜬 상태에서 들어오는 알림은 corner.
  //   - 정상 흐름 (사람 검출) 에선 addNotif() 가 applyMode 보다 먼저 호출돼
  //     이미 center 알림 발생 후 centerNextNotif=false 처리됨 → 여기선 no-op.
  //   - 디버그 버튼 클릭 시엔 addNotif 호출 없이 active 전환만 됨 →
  //     여기서 centerNextNotif=false 로 설정해 차후 알림이 corner 로 가게.
  if (!prevActiveForNotif && isActive) {
    centerNextNotif = false;
  }
  prevActiveForNotif = isActive;
  document.getElementById('standby').classList.toggle('hidden', isActive);
  // Dashboard UI 가시화
  document.getElementById('legend').classList.toggle('hidden-el', !isActive);
  document.getElementById('dashStats').classList.toggle('hidden-el', !isActive);
  document.getElementById('controls').classList.toggle('hidden-el', !isActive);
  // 디버그 배지
  document.getElementById('dsDebugBadge').classList.toggle('hidden-el', !manualMode);
}

// ── 시각 갱신 (standby clock) ──────────────────────────────────
function pad(n) { return n < 10 ? '0' + n : '' + n; }
function nowKr() {
  const d = new Date();
  const h12 = d.getHours() % 12 || 12;
  const ampm = d.getHours() < 12 ? '오전' : '오후';
  return `${ampm} ${h12}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}
setInterval(() => {
  document.getElementById('clockTime').textContent = nowKr();
}, 1000);

// 매 tick 마다 active 상태 재계산 (cooldown 만료 시 자동 standby 복귀).
setInterval(applyMode, 500);

// ── 알림 카드 — 동시 1개만 ──────────────────────────────────
// 첫 알림 (대기→활성 전환 시) = 화면 정중앙 (alert dialog 스타일).
// 그 이후 알림 (활성 상태에서 추가) = 우상단 (작게, 방해 적게).
// 활성→대기 전환되면 다시 다음 알림은 중앙으로.
let notifSeq = 0;
let centerNextNotif = true;        // 첫 알림은 항상 center
let prevActiveForNotif = false;    // 직전 frame 의 active 상태 (전환 감지용)

function addNotif(text) {
  const container = document.getElementById('notifs');
  // 이미 활성(닫힘 중 아님) 카드가 있으면 새 알림 무시 — 누적 방지.
  for (const c of container.children) {
    if (!c.classList.contains('closing')) return;
  }
  // 위치 결정 → container class 토글.
  const useCenter = centerNextNotif;
  centerNextNotif = false;   // 이번 알림 후로는 corner (다음 standby 까지)
  container.classList.toggle('center', useCenter);
  container.classList.toggle('corner', !useCenter);

  notifSeq++;
  const wrap = document.createElement('div');
  wrap.className = 'notif';
  wrap.innerHTML = `
    <div class="notif-top">
      <div class="notif-icon">&#x1F464;</div>
      <div class="notif-body">
        <div class="notif-head">
          <span class="notif-title">사람 감지</span>
          <div class="notif-time-row">
            <span class="notif-time">${nowKr()}</span>
            <button class="notif-close" aria-label="닫기">✕</button>
          </div>
        </div>
        <span class="notif-msg">${text}</span>
      </div>
    </div>
    <div class="notif-actions">
      <button class="notif-report">오탐 보고</button>
      <button class="notif-confirm">확인</button>
    </div>
  `;
  container.appendChild(wrap);
  const closeCard = () => {
    wrap.classList.add('closing');
    setTimeout(() => wrap.remove(), 400);
  };
  wrap.querySelector('.notif-close').addEventListener('click', closeCard);
  wrap.querySelector('.notif-confirm').addEventListener('click', closeCard);
  wrap.querySelector('.notif-report').addEventListener('click', () => {
    openReportModal();
    closeCard();
  });
}

// ── 오탐 보고 modal 로직 ─────────────────────────────────────
let reportSelectedLabel = null;
function openReportModal() {
  reportSelectedLabel = null;
  document.querySelectorAll('#reportChoices .report-choice').forEach(b => {
    b.classList.remove('selected');
  });
  document.getElementById('reportNote').value = '';
  document.getElementById('reportSubmit').disabled = true;
  document.getElementById('reportStatus').textContent = '';
  document.getElementById('reportModal').classList.add('show');
}
function closeReportModal() {
  document.getElementById('reportModal').classList.remove('show');
}
document.querySelectorAll('#reportChoices .report-choice').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#reportChoices .report-choice').forEach(b => b.classList.remove('selected'));
    btn.classList.add('selected');
    reportSelectedLabel = btn.getAttribute('data-label');
    document.getElementById('reportSubmit').disabled = false;
  });
});
document.getElementById('reportCancel').addEventListener('click', closeReportModal);
document.getElementById('reportSubmit').addEventListener('click', async () => {
  if (!reportSelectedLabel) return;
  const submitBtn = document.getElementById('reportSubmit');
  const status = document.getElementById('reportStatus');
  submitBtn.disabled = true;
  status.textContent = '저장 중...';
  try {
    const resp = await fetch('/report', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        label: reportSelectedLabel,
        note: document.getElementById('reportNote').value.slice(0, 1000),
      }),
    });
    const j = await resp.json();
    if (j.ok) {
      status.textContent = `✓ 저장됨 (${j.frame_count} frames) — ${j.report_id}`;
      status.style.color = '#1a7f1a';
      setTimeout(closeReportModal, 1500);
    } else {
      status.textContent = `실패: ${j.msg || 'unknown'}`;
      status.style.color = '#c52a2a';
      submitBtn.disabled = false;
    }
  } catch (e) {
    status.textContent = `에러: ${e.message}`;
    status.style.color = '#c52a2a';
    submitBtn.disabled = false;
  }
});

// ── WebSocket ──────────────────────────────────────────────────
let ws = null;
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  const statusEl = document.getElementById('connStatus');
  ws.onopen = () => {
    statusEl.textContent = '● 연결됨';
    statusEl.style.color = '#1a7f1a';
  };
  ws.onclose = () => {
    statusEl.textContent = '● 연결 끊김 - 재연결...';
    statusEl.style.color = '#c52a2a';
    setTimeout(connect, 1500);
  };
  ws.onerror = (e) => {
    console.error('[ws] error', e);
  };
  ws.onmessage = (ev) => {
    let d;
    try { d = JSON.parse(ev.data); } catch (err) { return; }
    handleFrame(d);
  };
}
connect();

// ── JPG 좌측 패널 갱신 (— body[data-has-jpg="true"] 일 때만 보임) ─
const HAS_JPG = document.body.dataset.hasJpg === "true";
const jpgFrame = document.getElementById('jpgFrame');
const jpgStatus = document.getElementById('jpgStatus');
let lastJpgName = "";
function updateJpg(d) {
  if (!HAS_JPG) return;
  const name = d.jpg || "";
  if (name && name !== lastJpgName) {
    jpgFrame.src = "/jpg/" + name;
    lastJpgName = name;
  } else if (!name && lastJpgName) {
    jpgFrame.removeAttribute('src');
    lastJpgName = "";
  }
  const t = d.T || 0;
  if (t > 0) jpgStatus.textContent = "T=" + t.toFixed(3) + (name ? "" : " (매칭 없음)");
  else       jpgStatus.textContent = "타임스탬프 없음";
}

// ── Frame handler ──────────────────────────────────────────────
function handleFrame(d) {
  const pts = d.pts || [];
  const tracks = d.tracks || [];
  const nowMs = performance.now();

  // 사람 분류 track id cache — TTL 기반. 한 번 사람으로 분류되면 N초간 유지.
  // 분류값이 잠시 떨어지거나 트래킹이 잠시 끊겨도 점 색이 안 깜빡임.
  // 같은 id 가 다시 사람으로 분류되면 TTL 갱신.
  for (const t of tracks) {
    const prob = t.human_prob !== undefined ? t.human_prob : 0;
    if (prob >= SEQ_THRESHOLD && t.id !== undefined) {
      personIdCache.set(t.id, nowMs + PERSON_COLOR_PERSIST_MS);
    }
  }
  // 만료된 entry 제거.
  for (const [tid, exp] of personIdCache) {
    if (nowMs > exp) personIdCache.delete(tid);
  }
  const personIds = new Set(personIdCache.keys());

  // 추론 대기 track id (human_prob < 0 sentinel) — 노랑 색칠 대상.
  // 모듈이 cluster 는 인식했지만 cascade 시퀀스(40프레임) 가 아직 안 찬 상태.
  // 보통 사람 등장 후 2초 이내. 이후엔 personIds 또는 일반 cluster 로 분류.
  const pendingIds = new Set();
  for (const t of tracks) {
    if (t.human_prob !== undefined && t.human_prob < 0 && t.id !== undefined) {
      pendingIds.add(t.id);
    }
  }

  // 점 cloud + edge 갱신 — 우선순위: pending(노랑) > person(초록) > cluster(파랑) > 노이즈.
  updatePoints(pts, personIds, pendingIds);
  buildClusterEdges(pts, personIds, pendingIds);

  // 좌측 카메라 JPG 갱신
  updateJpg(d);

  // 박스는 hide — 라벨만 유지 (추론 값 표시). 매 frame 위치 갱신.
  updateTrackTargets(tracks);

  // 화면 속도 (수신 fps)
  if (d.f !== lastFrame) {
    frameCount++;
    lastFrame = d.f;
    const elapsed = (performance.now() - frameWindowStart) / 1000;
    if (elapsed > 2) {
      currentFps = frameCount / elapsed;
      frameCount = 0;
      frameWindowStart = performance.now();
    }
  }
  const fpsStr = currentFps.toFixed(0) + '/초';
  document.getElementById('sbFps').textContent = fpsStr;
  document.getElementById('dsFps').textContent = fpsStr;

  // 사람 감지 → 자동 active 갱신 + 알림
  const personPresent = (d.person && d.person.present) || false;
  if (personPresent) {
    activeUntilMs = Date.now() + ACTIVE_HOLD_S * 1000;
    if (d.person.new) {
      const best = d.person.best || 0;
      const count = d.person.count || 1;
      const who = count > 1 ? `${count}명` : '사람이';
      addNotif(`${who} 감지되었습니다 · ${(best * 100).toFixed(0)}%`);
    }
  }
  applyMode();

  // 누적 알람 stats
  const events = (d.alarm && d.alarm.events) || 0;
  document.getElementById('sbEvents').textContent = events + '회';
  document.getElementById('dsEvents').textContent = events + '회';

  // 마지막 감지 경과
  const lastSeen = (d.alarm && d.alarm.last) || 0;
  let lastStr = '기록 없음';
  if (lastSeen > 0) {
    if (personPresent) {
      lastStr = '지금';
    } else {
      const elapsed = Date.now() / 1000 - lastSeen;
      if (elapsed < 60)        lastStr = `${Math.floor(elapsed)}초 전`;
      else if (elapsed < 3600) lastStr = `${Math.floor(elapsed / 60)}분 전`;
      else                     lastStr = `${Math.floor(elapsed / 3600)}시간 전`;
    }
  }
  document.getElementById('sbLast').textContent = lastStr;

  // 신호 상태
  const age = d.age !== undefined ? d.age : -1;
  let sig, col;
  if (age < 0)        { sig = '연결 대기'; col = '#bf6a00'; }
  else if (age < 1)   { sig = '실시간 연결'; col = '#1a7f1a'; }
  else if (age < 5)   { sig = Math.floor(age) + '초 지연'; col = '#bf6a00'; }
  else                { sig = '신호 끊김'; col = '#c52a2a'; }
  const sbSignalEl = document.getElementById('sbSignal');
  sbSignalEl.textContent = sig;
  sbSignalEl.style.color = col;
  document.getElementById('dsSignal').textContent = sig;

  // Dashboard 통계
  document.getElementById('dsCount').textContent = ((d.person && d.person.count) || 0) + '명';
  document.getElementById('dsPts').textContent = pts.length + '개';
}
</script>

</body>
</html>
"""


# ============ Entry ============

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retina-host", help="Retina 4SN 모듈 IP (실시간 모드)")
    ap.add_argument("--retina-port", type=int, default=29173,
                    help="JSON publisher 포트 (기본 29173)")
    ap.add_argument("--demo", action="store_true",
                    help="장비 없이 합성 데이터로 UI 확인")
    ap.add_argument("--demo-fps", type=int, default=20,
                    help="--demo 모드 frame rate")
    ap.add_argument("--web-host", default="0.0.0.0",
                    help="Flask 바인딩 (기본 0.0.0.0)")
    ap.add_argument("--web-port", type=int, default=8050)
    ap.add_argument("--refresh-ms", type=int, default=100,
                    help="WebSocket push 주기(ms). 기본 100 = 10Hz. "
                         "라즈베리/Tailscale 부하 큰 환경은 200~300 권장.")
    ap.add_argument("--max-points", type=int, default=2000,
                    help="frame 당 send 점 수 상한. 기본 2000.")
    ap.add_argument("--human-threshold", type=float, default=0.7,
                    help="사람 판정 threshold (cascade human_prob 기준). "
                         "기본 0.7 = box 색 초록 임계와 일치. 더 관대히 보고 싶으면 0.5.")
    ap.add_argument("--min-box-extent", type=float, default=0.3,
                    help="박스의 최대 축 길이 (m) 가 이 값 미만이면 ghost 로 보고 "
                         "human_prob=0 으로 demote. 기본 0.3. "
                         "0 = 필터 끔 (원본 모델 출력 그대로).")
    ap.add_argument("--jpg-dir", default="",
                    help="카메라 JPG 디렉토리 (`{timestamp}.jpg` 파일들). "
                         "지정 시 좌측 카메라 패널이 fd.timestamp 와 가장 가까운 JPG 자동 매칭.")
    args = ap.parse_args()

    global REFRESH_MS, MAX_POINTS, HUMAN_THRESHOLD, _HAS_JPG, MIN_BOX_EXTENT_M
    REFRESH_MS = args.refresh_ms
    MAX_POINTS = args.max_points
    HUMAN_THRESHOLD = args.human_threshold
    MIN_BOX_EXTENT_M = args.min_box_extent

    # ── JPG 인덱싱 + /jpg/<name> route ────────────────────────────
    if args.jpg_dir:
        if not os.path.isdir(args.jpg_dir):
            print(f"[error] --jpg-dir 가 디렉토리가 아님: {args.jpg_dir}")
            return 1
        n = build_jpg_index(args.jpg_dir)
        _HAS_JPG = n > 0
        if _HAS_JPG:
            @app.route("/jpg/<name>")
            def serve_jpg(name):
                # 인덱싱된 파일만 (path traversal 방지).
                for ts in _jpg_ts_sorted:
                    p = _jpg_by_ts.get(ts)
                    if p and os.path.basename(p) == name:
                        return send_file(p, mimetype="image/jpeg")
                abort(404)

    if args.demo:
        print(f"[mode] DEMO ({args.demo_fps} fps 합성 데이터)")
        t = threading.Thread(target=demo_loop, args=(args.demo_fps,), daemon=True)
    else:
        if not args.retina_host:
            ap.error("실시간 모드는 --retina-host 필요 (또는 --demo)")
        t = threading.Thread(target=tcp_loop,
                             args=(args.retina_host, args.retina_port),
                             daemon=True)
    t.start()

    print(f"[web] http://<this-host>:{args.web_port}")
    print(f"[ws]  push every {REFRESH_MS}ms · max {MAX_POINTS} points · "
          f"human threshold {HUMAN_THRESHOLD}")
    # threaded=True → 각 WebSocket 클라이언트가 별도 스레드.
    app.run(host=args.web_host, port=args.web_port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
