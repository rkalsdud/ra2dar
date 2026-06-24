"""
SRS Retina 4SN 학습 데이터 수집기.

실시간으로 레이다 데이터를 받아 프레임당 JSON 파일을 저장합니다.

frame JSON (T, C, V, P, TID, TRK):
  - cluster_process.py로 전처리 가능
  - convert_to_supervisely.py로 라벨 변환 가능
  - 학습 dataloader가 직접 읽음

출력 폴더 구조:
  OUTPUT_BASE/
    └── <session_name>_frames/
          ├── <timestamp>.json
          └── ...

* radar_client.py의 parse_packet을 재사용합니다 (같은 폴더에 있어야 함).
"""

import os
import json
import time
import socket
from datetime import datetime

# radar_client.py의 파서 재사용
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from radar_client import parse_packet

# ====== 설정 (필요시 여기만 수정) ======
RADAR_IP = "192.168.30.1"
RADAR_PORT = 29172

# 출력 경로
OUTPUT_BASE = "./recorded_data"
SESSION_NAME = None      # None이면 자동: session_YYYYmmdd_HHMMSS

# JSON 들여쓰기 (기존 frame JSON과 동일)
JSON_INDENT = 4

# 연결 안정성
RECV_TIMEOUT_SEC = 10.0
RECONNECT_DELAY_SEC = 2.0
REPORT_INTERVAL_SEC = 1.0
# =====================================


def save_frame(frame, frames_dir):
    """한 frame을 JSON 형식으로 저장. 확장 필드(_*)는 제외.

    원자적 쓰기: *.tmp로 먼저 쓴 뒤 os.replace로 최종 파일명으로 교체.
    쓰기 도중 중단돼도 깨진 부분 파일은 *.tmp로만 남고 최종 파일은
    온전하거나 아예 존재하지 않음 — JSON 파싱 오류 영구 방지.
    """
    ts = f"{frame['T']:.6f}"
    save_data = {k: v for k, v in frame.items() if not k.startswith("_")}
    final_path = os.path.join(frames_dir, f"{ts}.json")
    tmp_path = final_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=JSON_INDENT, ensure_ascii=False)
    os.replace(tmp_path, final_path)


def main():
    print(f"📡 SRS Retina 4SN 학습 데이터 수집기")
    print(f"   대상 : {RADAR_IP}:{RADAR_PORT}\n")

    # 출력 폴더 준비
    session = SESSION_NAME or datetime.now().strftime("session_%Y%m%d_%H%M%S")
    frames_dir = os.path.join(OUTPUT_BASE, session + "_frames")
    os.makedirs(frames_dir, exist_ok=True)
    print(f"💾 JSON → {frames_dir}\n")

    total = 0
    while True:
        sock = None
        try:
            # TCP 접속
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(RECV_TIMEOUT_SEC)
            print(f"⏳ 접속 시도 → {RADAR_IP}:{RADAR_PORT}")
            sock.connect((RADAR_IP, RADAR_PORT))
            print("✅ 연결 성공 — Ctrl+C로 종료\n")
            sock.settimeout(None)

            fps_count = 0
            last_report = time.time()
            last_n_pts = 0
            last_tracks = []

            while True:
                frame = parse_packet(sock)
                if frame is None:
                    continue

                total += 1
                fps_count += 1
                last_n_pts = len(frame["C"]) // 3
                last_tracks = frame.get("_tracks_all", [])

                # 저장
                save_frame(frame, frames_dir)

                # 주기적 리포트
                now = time.time()
                if now - last_report >= REPORT_INTERVAL_SEC:
                    fps = fps_count / (now - last_report)
                    status_list = [t["status_name"] for t in last_tracks]
                    status_str = ", ".join(status_list) if status_list else "—"
                    print(f"📊 #{total:>6}  {fps:5.1f} fps  "
                          f"점 {last_n_pts:>4d}  트랙 {len(last_tracks)}  [{status_str}]")
                    fps_count = 0
                    last_report = now

        except KeyboardInterrupt:
            print(f"\n🛑 수집 종료. 총 {total} frame 저장")
            print(f"   → {frames_dir}")
            print(f"\n다음 단계 예시:")
            print(f"   cluster_process.py     로 전처리")
            print(f"   convert_to_supervisely 로 라벨 생성")
            break
        except (ConnectionError, socket.timeout, OSError) as e:
            print(f"⚠️  연결 문제: {e}. {RECONNECT_DELAY_SEC}초 후 재시도\n")
            time.sleep(RECONNECT_DELAY_SEC)
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
