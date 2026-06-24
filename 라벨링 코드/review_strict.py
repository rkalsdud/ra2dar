#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""review_strict.py — Strict JSONL + dataset zip viewer (v19)

`workbench_outputs/<scenario>/labels_final_all_frames_schema.jsonl` 와
원본 dataset zip (예: 512stand.zip) 을 함께 띄워서 frame 별 point cloud +
auto-label box 를 3D 로 보면서 빠르게 검토할 수 있게 만든 도구.

기본 사용
    python3 review_strict.py
        → 같은 폴더의 512stand.zip + workbench_outputs/512stand/* 자동 로드

옵션
    --zip 512stand.zip
    --labels workbench_outputs/512stand/labels_final_all_frames_schema.jsonl
    --quality workbench_outputs/512stand/auto_box_quality.csv
    --suspect-only           ← auto_box_quality.csv 에서 suspect=yes 인 frame 만
    --start 1234             ← 시작 frame index
    --no-empty               ← empty (objects=[]) frame은 건너뜀

키보드
    ← / p        이전 frame
    → / n        다음 frame
    space        재생/일시정지
    s            현재 frame을 review_decisions.csv 에 'ok' 로 기록
    r            'review' 로 기록 (의심 frame 표시)
    j            jump dialog (frame index 직접 입력)
    f            'suspect=yes' 인 다음 frame 으로 점프
    q / esc      종료
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, TextBox, Button
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
except Exception as e:
    print("matplotlib이 필요합니다. `pip3 install matplotlib` 후 다시 실행하세요.", file=sys.stderr)
    print(f"  detail: {e}", file=sys.stderr)
    sys.exit(1)

# 동일 폴더의 workbench 모듈에서 canonical parse_points/is_raw_frame_json 을 가져온다.
# 이 데이터셋은 frame json이 C/V/P/TID 같은 압축 키를 쓰기 때문에 자체 파서로는 못 읽음.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
try:
    from radar_temporal_label_workbench import parse_points as _wb_parse_points, is_raw_frame_json as _wb_is_raw_frame_json  # type: ignore
    _USE_WB_PARSER = True
except Exception:
    _USE_WB_PARSER = False


def find_default_zip(here: Path) -> Optional[Path]:
    zips = [p for p in here.glob("*.zip") if not p.name.startswith("radar_temporal_label_workbench")]
    if not zips:
        return None
    return sorted(zips, key=lambda p: p.stat().st_size, reverse=True)[0]


def parse_points(frame: Dict[str, Any]) -> List[Tuple[float, float, float, float, float, int]]:
    """원본 frame json → list of (x,y,z,velocity,power,tid). radar_temporal_label_workbench.parse_points 와 동일 인터페이스."""
    if _USE_WB_PARSER:
        try:
            return list(_wb_parse_points(frame))
        except Exception:
            pass
    pts: List[Tuple[float, float, float, float, float, int]] = []
    # 가장 흔한 포맷: {"points": [{"x":..,"y":..,"z":..,...}, ...]}
    raw = frame.get("points") or frame.get("point_cloud") or frame.get("pc") or []
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        for p in raw:
            try:
                pts.append((
                    float(p.get("x", 0.0)),
                    float(p.get("y", 0.0)),
                    float(p.get("z", 0.0)),
                    float(p.get("v") or p.get("velocity") or 0.0),
                    float(p.get("p") or p.get("power") or p.get("intensity") or 0.0),
                    int(p.get("tid") or p.get("track_id") or 255),
                ))
            except Exception:
                continue
        return pts
    # array-of-array 포맷: {"points": [[x,y,z], ...]}
    if isinstance(raw, list) and raw and isinstance(raw[0], (list, tuple)):
        for row in raw:
            if len(row) >= 3:
                try:
                    pts.append((float(row[0]), float(row[1]), float(row[2]),
                                float(row[3]) if len(row) > 3 else 0.0,
                                float(row[4]) if len(row) > 4 else 0.0,
                                int(row[5]) if len(row) > 5 else 255))
                except Exception:
                    continue
        return pts
    # parallel array 포맷: {"x":[...],"y":[...],"z":[...]}
    if isinstance(frame.get("x"), list):
        xs = frame.get("x") or []
        ys = frame.get("y") or []
        zs = frame.get("z") or []
        vs = frame.get("v") or frame.get("velocity") or []
        ps_ = frame.get("p") or frame.get("power") or []
        tids = frame.get("tid") or frame.get("track_id") or []
        n = min(len(xs), len(ys), len(zs))
        for i in range(n):
            try:
                pts.append((
                    float(xs[i]), float(ys[i]), float(zs[i]),
                    float(vs[i]) if i < len(vs) else 0.0,
                    float(ps_[i]) if i < len(ps_) else 0.0,
                    int(tids[i]) if i < len(tids) else 255,
                ))
            except Exception:
                continue
    return pts


def is_raw_frame_json(frame: Any) -> bool:
    if _USE_WB_PARSER:
        try:
            return bool(_wb_is_raw_frame_json(frame))
        except Exception:
            pass
    return isinstance(frame, dict) and (
        isinstance(frame.get("points"), list)
        or isinstance(frame.get("point_cloud"), list)
        or isinstance(frame.get("pc"), list)
        or isinstance(frame.get("x"), list)
        or isinstance(frame.get("C"), list)
    )


def frame_id_from_name(name: str) -> str:
    base = name.split("/")[-1]
    if base.endswith(".json"):
        base = base[:-5]
    return base


def record_from_name(name: str) -> str:
    parts = name.split("/")
    return parts[-2] if len(parts) >= 2 else ""


def source_priority(path_name: str) -> Tuple[int, str]:
    low = path_name.lower()
    parts = [p.lower() for p in path_name.split("/")]
    if any(p.endswith("_frames") or p == "_frames" for p in parts):
        return (0, path_name)
    if "preprocessed" in low or "processed" in low:
        return (1, path_name)
    return (2, path_name)


def build_zip_index(zip_path: Path) -> Dict[str, str]:
    """frame_id → archive name (dedup, _frames priority)."""
    index: Dict[str, Tuple[Tuple[int, str], str]] = {}
    with zipfile.ZipFile(zip_path, "r") as z:
        for n in z.namelist():
            if not n.endswith(".json") or n.startswith("__MACOSX/"):
                continue
            fid = frame_id_from_name(n)
            pr = source_priority(n)
            if fid not in index or pr < index[fid][0]:
                index[fid] = (pr, n)
    return {fid: name for fid, (_, name) in index.items()}


def load_labels(path: Path) -> List[Dict[str, Any]]:
    labels: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                labels.append(json.loads(line))
            except Exception:
                pass
    return labels


def load_quality(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["frame_id"]] = row
    return out


def box_edges(box: Dict[str, Any]) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """strict box {center,dimensions,yaw} → list of 12 edges."""
    import math
    cx, cy, cz = (float(v) for v in box.get("center", [0.0, 0.0, 0.0]))
    dx, dy, dz = (float(v) for v in box.get("dimensions", [0.8, 0.8, 1.6]))
    yaw = float(box.get("yaw", 0.0))
    hx, hy, hz = dx / 2, dy / 2, dz / 2
    corners_local = [
        (-hx, -hy, -hz), (+hx, -hy, -hz), (+hx, +hy, -hz), (-hx, +hy, -hz),
        (-hx, -hy, +hz), (+hx, -hy, +hz), (+hx, +hy, +hz), (-hx, +hy, +hz),
    ]
    ca, sa = math.cos(yaw), math.sin(yaw)
    corners = []
    for lx, ly, lz in corners_local:
        wx = lx * ca - ly * sa + cx
        wy = lx * sa + ly * ca + cy
        wz = lz + cz
        corners.append((wx, wy, wz))
    edges_idx = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return [(corners[a], corners[b]) for a, b in edges_idx]


class Reviewer:
    def __init__(
        self,
        zip_path: Path,
        labels: List[Dict[str, Any]],
        zip_index: Dict[str, str],
        quality: Dict[str, Dict[str, Any]],
        order: List[int],
        decisions_path: Path,
        title_prefix: str,
    ):
        self.zip_path = zip_path
        self.labels = labels
        self.zip_index = zip_index
        self.quality = quality
        self.order = order
        self.decisions_path = decisions_path
        self.title_prefix = title_prefix
        self.idx = 0
        self.playing = False
        self.last_play_tick = 0.0
        self.decisions: Dict[str, str] = self._load_decisions()
        self.zip_handle = zipfile.ZipFile(zip_path, "r")
        self.fig = plt.figure(figsize=(12, 9))
        # 3D 메인 영역. 아래쪽에 widget bar 자리를 남겨둠.
        self.ax = self.fig.add_axes([0.05, 0.18, 0.9, 0.78], projection="3d")
        # ---- 하단 navigation widgets ----
        # 슬라이더: 1..len(order) 의 frame index 직접 선택
        n = max(1, len(self.order))
        slider_ax = self.fig.add_axes([0.08, 0.08, 0.62, 0.03])
        self.slider = Slider(slider_ax, "frame", 1, n, valinit=1, valstep=1, valfmt="%d")
        self.slider.on_changed(self._on_slider)
        # 텍스트박스: index 또는 frame_id 직접 입력
        tb_ax = self.fig.add_axes([0.78, 0.08, 0.16, 0.04])
        self.textbox = TextBox(tb_ax, "go to", initial="")
        self.textbox.on_submit(self._on_textbox_submit)
        # 이전/다음/play/suspect 버튼
        btn_prev_ax = self.fig.add_axes([0.08, 0.02, 0.08, 0.04])
        btn_next_ax = self.fig.add_axes([0.17, 0.02, 0.08, 0.04])
        btn_play_ax = self.fig.add_axes([0.26, 0.02, 0.08, 0.04])
        btn_susp_ax = self.fig.add_axes([0.35, 0.02, 0.12, 0.04])
        btn_ok_ax   = self.fig.add_axes([0.48, 0.02, 0.08, 0.04])
        btn_rev_ax  = self.fig.add_axes([0.57, 0.02, 0.10, 0.04])
        self.btn_prev = Button(btn_prev_ax, "◀ prev")
        self.btn_next = Button(btn_next_ax, "next ▶")
        self.btn_play = Button(btn_play_ax, "play")
        self.btn_susp = Button(btn_susp_ax, "next-suspect")
        self.btn_ok   = Button(btn_ok_ax,   "ok (s)")
        self.btn_rev  = Button(btn_rev_ax,  "review (r)")
        self.btn_prev.on_clicked(lambda _e: self._step(-1))
        self.btn_next.on_clicked(lambda _e: self._step(+1))
        self.btn_play.on_clicked(lambda _e: self._toggle_play())
        self.btn_susp.on_clicked(lambda _e: self._jump_to_next_suspect())
        self.btn_ok.on_clicked(lambda _e: self._mark("ok"))
        self.btn_rev.on_clicked(lambda _e: self._mark("review"))
        # 슬라이더가 _render() 안에서 set_val 호출될 때 재귀로 _on_slider 가 안 돌게 막는 플래그
        self._slider_internal_update = False
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", lambda _e: self._save_decisions())
        # plotting limits — radar 좌표계 기본값. 데이터에 맞춰 자동으로도 조정됨.
        self.x_lim = (-5, 5)
        self.y_lim = (0, 10)
        self.z_lim = (-2.0, 2.0)

    def _load_decisions(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        if self.decisions_path.exists():
            with self.decisions_path.open("r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    out[row["frame_id"]] = row.get("decision", "")
        return out

    def _save_decisions(self) -> None:
        with self.decisions_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["frame_id", "decision", "saved_at"])
            w.writeheader()
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            for fid, dec in sorted(self.decisions.items()):
                w.writerow({"frame_id": fid, "decision": dec, "saved_at": stamp})

    def _frame_payload(self, label_idx: int) -> Tuple[Dict[str, Any], List[Tuple[float, float, float, float, float, int]]]:
        lab = self.labels[label_idx]
        fid = lab.get("frame_id", "")
        zname = self.zip_index.get(fid)
        if not zname:
            return lab, []
        try:
            data = json.loads(self.zip_handle.read(zname))
        except Exception:
            return lab, []
        pts = parse_points(data) if is_raw_frame_json(data) else []
        return lab, pts

    def _render(self) -> None:
        order_pos = self.idx
        if not self.order:
            self.ax.clear()
            self.ax.set_title("표시할 frame 없음")
            self.fig.canvas.draw_idle()
            return
        label_idx = self.order[order_pos]
        lab, pts = self._frame_payload(label_idx)
        fid = lab.get("frame_id", "")
        q = self.quality.get(fid, {})
        objects = lab.get("objects", []) or []

        self.ax.clear()
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")

        if pts:
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
            self.ax.scatter(xs, ys, zs, c=zs, cmap="coolwarm", s=4, depthshade=False, alpha=0.75)
            mx, Mx = min(min(xs), -1), max(max(xs), 1)
            my, My = min(min(ys), 0), max(max(ys), 5)
            mz, Mz = min(min(zs), -2), max(max(zs), 2)
            pad = 0.5
            self.x_lim = (mx - pad, Mx + pad)
            self.y_lim = (my - pad, My + pad)
            self.z_lim = (mz - pad, Mz + pad)
        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)
        self.ax.set_zlim(self.z_lim)

        for obj in objects:
            box = obj.get("box") or {}
            for (a, b) in box_edges(box):
                self.ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="#e53935", linewidth=1.4)
            if box.get("center"):
                cx, cy, cz = box["center"]
                self.ax.text(cx, cy, cz, f" {obj.get('class','?')}/{obj.get('pose','?')}", color="#e53935", fontsize=9)

        suspect = (q.get("suspect") == "yes")
        decision = self.decisions.get(fid, "")
        suspect_str = f"  [SUSPECT] {q.get('suspect_reason','')}" if suspect else ""
        decision_str = f"  [decision={decision}]" if decision else ""
        playing = "PLAYING" if self.playing else "paused"
        title = (
            f"{self.title_prefix}  [{order_pos+1}/{len(self.order)}]  fid={fid}\n"
            f"pts={q.get('n_points','?')}  objs={len(objects)}  "
            f"box_z=[{q.get('box_bottom_z','-')}, {q.get('box_top_z','-')}]  "
            f"z_bot_off={q.get('z_bottom_offset','-')}  z_top_off={q.get('z_top_offset','-')}  xy_off={q.get('xy_offset','-')}"
            f"{suspect_str}{decision_str}\n"
            f"{playing}  (arrows/n/p navigate, space play, s=ok, r=review, f=next-suspect, j=jump, q=quit)"
        )
        self.ax.set_title(title, fontsize=10)
        # 슬라이더 위치 동기화 (재귀 콜백 방지)
        try:
            self._slider_internal_update = True
            self.slider.set_val(order_pos + 1)
        finally:
            self._slider_internal_update = False
        # play 버튼 라벨 토글
        if hasattr(self, "btn_play"):
            self.btn_play.label.set_text("pause" if self.playing else "play")
        self.fig.canvas.draw_idle()

    # ---- widget callbacks ----
    def _step(self, delta: int) -> None:
        if not self.order:
            return
        self.idx = (self.idx + delta) % len(self.order)
        self.playing = False
        self._render()

    def _toggle_play(self) -> None:
        self.playing = not self.playing
        if self.playing:
            self.last_play_tick = time.time()
            self._play_loop()
        else:
            self._render()

    def _on_slider(self, val) -> None:
        if self._slider_internal_update:
            return
        target = int(val) - 1
        target = max(0, min(len(self.order) - 1, target))
        if target == self.idx:
            return
        self.idx = target
        self.playing = False
        self._render()

    def _on_textbox_submit(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        # 숫자면 1-based index, 아니면 frame_id 로 검색
        if text.isdigit():
            target = max(0, min(len(self.order) - 1, int(text) - 1))
            self.idx = target
        else:
            for i, lab_idx in enumerate(self.order):
                if self.labels[lab_idx].get("frame_id") == text:
                    self.idx = i
                    break
            else:
                # frame_id 정확 일치 안하면 부분 매칭 시도
                for i, lab_idx in enumerate(self.order):
                    fid = self.labels[lab_idx].get("frame_id", "")
                    if text in fid:
                        self.idx = i
                        break
        self.playing = False
        # 입력 박스 비우기 (다음 입력 받기 좋게)
        try:
            self.textbox.set_val("")
        except Exception:
            pass
        self._render()

    def _on_key(self, event) -> None:
        if event.key in ("left", "p"):
            self.idx = (self.idx - 1) % max(1, len(self.order))
            self.playing = False
            self._render()
        elif event.key in ("right", "n"):
            self.idx = (self.idx + 1) % max(1, len(self.order))
            self.playing = False
            self._render()
        elif event.key == " ":
            self.playing = not self.playing
            if self.playing:
                self.last_play_tick = time.time()
                self._play_loop()
            else:
                self._render()
        elif event.key == "s":
            self._mark("ok")
        elif event.key == "r":
            self._mark("review")
        elif event.key == "f":
            self._jump_to_next_suspect()
        elif event.key == "j":
            self._prompt_jump()
        elif event.key in ("q", "escape"):
            plt.close(self.fig)

    def _play_loop(self) -> None:
        if not self.playing:
            return
        self.idx = (self.idx + 1) % max(1, len(self.order))
        self._render()
        self.fig.canvas.start_event_loop(0.10)
        if plt.fignum_exists(self.fig.number) and self.playing:
            self._play_loop()

    def _mark(self, value: str) -> None:
        if not self.order:
            return
        fid = self.labels[self.order[self.idx]].get("frame_id", "")
        if fid:
            self.decisions[fid] = value
            self._save_decisions()
        self.idx = (self.idx + 1) % max(1, len(self.order))
        self._render()

    def _jump_to_next_suspect(self) -> None:
        n = len(self.order)
        if not n:
            return
        for k in range(1, n + 1):
            cand = (self.idx + k) % n
            fid = self.labels[self.order[cand]].get("frame_id", "")
            if self.quality.get(fid, {}).get("suspect") == "yes":
                self.idx = cand
                self.playing = False
                self._render()
                return

    def _prompt_jump(self) -> None:
        try:
            txt = input("jump to index (1-based) or frame_id: ").strip()
        except EOFError:
            return
        if not txt:
            return
        if txt.isdigit():
            self.idx = max(0, min(len(self.order) - 1, int(txt) - 1))
        else:
            for i, lab_idx in enumerate(self.order):
                if self.labels[lab_idx].get("frame_id") == txt:
                    self.idx = i
                    break
        self.playing = False
        self._render()

    def run(self) -> None:
        self._render()
        plt.show()
        self._save_decisions()
        try:
            self.zip_handle.close()
        except Exception:
            pass


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--zip", default=None, help="dataset zip path. 미지정시 같은 폴더의 가장 큰 zip 사용")
    ap.add_argument("--labels", default=None, help="strict JSONL 경로. 미지정시 workbench_outputs/<scenario>/labels_final_all_frames_schema.jsonl")
    ap.add_argument("--quality", default=None, help="auto_box_quality.csv 경로. 미지정시 자동 추정")
    ap.add_argument("--decisions", default=None, help="기록 CSV 경로. 미지정시 workbench_outputs/<scenario>/review_decisions.csv")
    ap.add_argument("--suspect-only", action="store_true", help="auto_box_quality.csv 에서 suspect=yes 인 frame 만 본다")
    ap.add_argument("--no-empty", action="store_true", help="objects=[] frame 은 건너뛴다")
    ap.add_argument("--start", type=int, default=1, help="시작 frame index (1-based)")
    args = ap.parse_args()

    zip_path = Path(args.zip).resolve() if args.zip else find_default_zip(here)
    if not zip_path or not zip_path.exists():
        print(f"dataset zip을 찾을 수 없음: {zip_path}", file=sys.stderr); sys.exit(1)
    scenario = zip_path.stem
    out_dir = here / "workbench_outputs" / scenario
    labels_path = Path(args.labels).resolve() if args.labels else out_dir / "labels_final_all_frames_schema.jsonl"
    if not labels_path.exists():
        print(f"strict JSONL을 찾을 수 없음: {labels_path}", file=sys.stderr)
        print("먼저 radar_temporal_label_workbench.py 로 Load + Export 를 한 번 돌리세요.", file=sys.stderr)
        sys.exit(1)
    quality_path = Path(args.quality).resolve() if args.quality else out_dir / "auto_box_quality.csv"
    decisions_path = Path(args.decisions).resolve() if args.decisions else out_dir / "review_decisions.csv"

    print(f"zip:       {zip_path}")
    print(f"labels:    {labels_path}")
    print(f"quality:   {quality_path} ({'found' if quality_path.exists() else 'missing'})")
    print(f"decisions: {decisions_path}")

    print("loading frame index from zip ...", end="", flush=True)
    zip_index = build_zip_index(zip_path)
    print(f" {len(zip_index)} frames")
    print("loading labels ...", end="", flush=True)
    labels = load_labels(labels_path)
    print(f" {len(labels)} rows")
    quality = load_quality(quality_path)
    print(f"quality rows: {len(quality)}")

    order: List[int] = []
    for i, lab in enumerate(labels):
        fid = lab.get("frame_id", "")
        if args.no_empty and not (lab.get("objects") or []):
            continue
        if args.suspect_only and quality.get(fid, {}).get("suspect") != "yes":
            continue
        if fid not in zip_index:
            continue
        order.append(i)
    if not order:
        print("표시할 frame이 없음. --suspect-only 옵션을 끄고 다시 시도해보세요.", file=sys.stderr)
        sys.exit(2)
    print(f"will display {len(order)} frames"
          + (" (suspect-only)" if args.suspect_only else "")
          + (" (no-empty)" if args.no_empty else ""))

    title_prefix = f"[{scenario}]"
    reviewer = Reviewer(zip_path, labels, zip_index, quality, order, decisions_path, title_prefix)
    reviewer.idx = max(0, min(len(order) - 1, args.start - 1))
    reviewer.run()


if __name__ == "__main__":
    main()
