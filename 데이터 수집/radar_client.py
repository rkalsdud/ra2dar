"""
SRS Retina 4SN 레이다 실시간 TCP 클라이언트.

레이다(TCP:29172)에 직접 접속해 매 frame을 파싱하고,
기존 frame JSON 형식(T, C, V, P, TID, TRK)으로 저장 또는 실시간 처리합니다.

참고 프로토콜:
  examples/embedded/retinaoutputexample/pointcloudsrecv/pointcloudsrecv.cpp

세 가지 모드 지원 (동시 사용 가능):
  1) SAVE_JSON           — frame JSON 파일 저장 (학습 데이터 수집용)
  2) APPLY_PROCESSING    — cluster_process.process_frame 로 즉시 전처리
                           (filter+voxel+SOR 까지. cluster 분리 안 함)
  3) STREAM_MODEL_INPUT  — cluster_process.process_frame_to_model_input 으로
                           per-cluster (N, 5) 정규화 입력까지 만들어
                           on_model_input(...) 콜백으로 즉시 넘김.
                           파일 저장 없음. 모델 추론 / ROS publish 에 직접 연결.
"""

import os
import json
import time
import socket
import struct
import numpy as np
from datetime import datetime

# ====== 설정 (필요시 여기만 수정) ======
RADAR_IP = "192.168.30.1"
RADAR_PORT = 29172

# 모드 토글 (셋 다 True 가능 — 동시 진행)
SAVE_JSON = False                # frame JSON 파일로 저장 (학습 데이터 수집)
APPLY_PROCESSING = False        # cluster_process.process_frame (cluster 분리 X)
STREAM_MODEL_INPUT = False      # cluster_process.process_frame_to_model_input
                                # (per-cluster (N, 5) 입력까지 만들어 콜백 호출)

# 저장 설정
OUTPUT_BASE = "./"
SESSION_NAME = None            # None이면 시간 기반 자동 생성: session_YYYYmmdd_HHMMSS

# 연결 안정성
RECV_TIMEOUT_SEC = 10.0
RECONNECT_DELAY_SEC = 2.0

# 진단 출력 주기
REPORT_INTERVAL_SEC = 1.0

# STREAM_MODEL_INPUT 모드에서 사용할 콜백.
#
# 종속 관계:
#   cluster_process.process_frame_to_model_input(...)  ← 단일 진실 소스
#         ↓                              ↓
#   make_model_input.py             radar_client.py
#   (오프라인 배치, 파일 저장)        (실시간 stream, 콜백)
#
# cluster_process.py 상단의 MODEL_N_POINTS, DBSCAN_EPS 등을 바꾸면
# 위 두 곳 모두 자동으로 같은 값을 쓰게 됨.
#
# 모델 학습이 끝난 뒤 실제 추론에 연결하려면 아래 set_model_input_callback 으로
# 교체. 그러면 매 frame 마다 (B, 256, 5) 텐서가 콜백에 전달됨.
def _default_on_model_input(clusters, frame_meta):
    """기본 콜백 — frame 당 cluster 개수와 모델 입력 shape 만 print.

    Args:
        clusters: List[cluster_process.ClusterInput]
                  각 cluster: .points (MODEL_N_POINTS, 5), .cluster_id,
                              .centroid_xy, .z_min, .scale, .n_orig_points
        frame_meta: dict — {"T", "frame_count", "n_orig_points", "tracks": [...]}

    실제 모델을 붙일 때 참고할 코드:
        import cluster_process as cp
        import torch
        X, meta = cp.stack_model_input(clusters)      # (B, 256, 5) numpy
        X_t = torch.from_numpy(X).to(device)
        with torch.no_grad():
            pred = model(X_t)
        # pred + meta(centroid_xy/z_min/scale) 로 박스 절대좌표 복원 후 사용
    """
    import cluster_process as cp_local
    X, meta = cp_local.stack_model_input(clusters)
    print(f"   ⤷ 모델 입력 batch: {X.shape} (cluster {len(clusters)}개, "
          f"T={frame_meta['T']:.3f})")
    # 디버그용: 첫 cluster 의 절대 위치
    if meta:
        cx, cy = meta[0]["centroid_xy"]
        print(f"      ↳ cluster #{meta[0]['cluster_id']}: "
              f"centroid=({cx:+.2f}, {cy:+.2f}), n_orig={meta[0]['n_orig_points']}")


on_model_input = _default_on_model_input


def set_model_input_callback(fn):
    """STREAM_MODEL_INPUT 콜백 교체.

    추론 모듈을 별도 파일에 두고 다음과 같이 연결:
        from src import radar_client, cluster_process
        import torch

        model = ...   # 학습된 PointNet++ 모델
        device = torch.device("cuda")

        def on_input(clusters, meta):
            X, info = cluster_process.stack_model_input(clusters)
            X_t = torch.from_numpy(X).to(device)
            with torch.no_grad():
                pred = model(X_t)
            # pred 박스를 info 로 절대좌표 복원해서 표시 / ROS publish

        radar_client.STREAM_MODEL_INPUT = True
        radar_client.SAVE_JSON = False     # 파일 저장 안 함
        radar_client.set_model_input_callback(on_input)
        radar_client.main()
    """
    global on_model_input
    on_model_input = fn
# =====================================

# 프로토콜 상수 (radar_packet.hpp, pointcloudsrecv.cpp 참고)
NETWORK_RX_MAGIC = bytes([0x21, 0x43, 0xCD, 0xAB])   # 0xABCD4321 LE
POINT_MAGIC_WORD = bytes([1, 2, 3, 4, 5, 6, 7, 8])   # 패킷 본문 매직워드
NETWORK_RX_HEADER_SIZE = 36
POINT_SIZE = 20       # x,y,z,doppler,power = 5 × float
TARGET_SIZE = 28      # posX,posY,status,id,reserved[3]

TARGET_STATUS_NAMES = {
    0: "STANDING",
    1: "LYING",
    2: "SITTING",
    3: "FALL",
    4: "UNKNOWN",
}


# ----- TCP 수신 유틸 -----

def recv_exact(sock, n):
    """정확히 n 바이트를 받아 반환. EOF면 ConnectionError."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("소켓이 닫혔습니다")
        buf.extend(chunk)
    return bytes(buf)


# ----- 패킷 파서 -----

def parse_packet(sock):
    """
    한 패킷을 수신·파싱.
    반환: 성공 시 frame_dict, 실패 시 None.

    frame_dict 키:
      T            : 수신 시각 (Unix epoch, 초)
      frame_count  : 레이다 자체 프레임 카운터
      C            : [x1,y1,z1, x2,y2,z2, ...]  (flat list)
      V            : [v1, v2, ...]              (doppler, m/s)
      P            : [p1, p2, ...]              (power)
      TID          : [tid1, tid2, ...]          (255 = 미할당)
      TRK          : [id, status, posX, posY]   (첫 트랙, 기존 형식 호환)
      _tracks_all  : [{id, status, status_name, posX, posY}, ...]  (확장)
    """
    # 1) 네트워크 헤더 36 B
    header = recv_exact(sock, NETWORK_RX_HEADER_SIZE)
    if header[4:8] != NETWORK_RX_MAGIC:
        return None  # 동기화 실패

    payload_size = struct.unpack_from("<I", header, 16)[0]
    if payload_size <= 0 or payload_size > 10_000_000:
        return None

    # 2) 페이로드
    body = recv_exact(sock, payload_size)

    # 3) 포인트 블록 매직워드 확인
    if body[:8] != POINT_MAGIC_WORD:
        return None

    frame_count, point_num = struct.unpack_from("<II", body, 8)
    points_start = 16
    points_end = points_start + point_num * POINT_SIZE
    if points_end > len(body):
        return None

    # 점 데이터 한 번에 numpy로 unpack (x, y, z, doppler, power)
    if point_num > 0:
        raw = struct.unpack_from(f"<{point_num * 5}f", body, points_start)
        pts = np.array(raw, dtype=np.float32).reshape(point_num, 5)
        coords = pts[:, :3].astype(np.float64)
        velocities = pts[:, 3].astype(np.float64)
        powers = pts[:, 4].astype(np.float64)
    else:
        coords = np.zeros((0, 3), dtype=np.float64)
        velocities = np.zeros(0, dtype=np.float64)
        powers = np.zeros(0, dtype=np.float64)

    # 4) 타겟 블록 위치 탐색 (다음 magic word 위치를 찾음)
    tids = np.full(point_num, 255, dtype=np.int64)  # 기본값 255 (미할당)
    tracks_all = []
    primary_trk = []

    if point_num > 0:
        # 가능한 위치 후보 (1B/점, 4B/점, 또는 고정 오프셋 48020)
        candidates = [
            points_end + point_num,
            points_end + point_num * 4,
            48020,
        ]
        target_magic_offset = None
        for off in candidates:
            if 0 <= off and off + 8 <= len(body) and body[off:off + 8] == POINT_MAGIC_WORD:
                target_magic_offset = off
                break

        if target_magic_offset is not None:
            # TID 추출 (보통 1바이트로 충분한 값)
            bytes_per_tid = max(1, (target_magic_offset - points_end) // point_num)
            for i in range(point_num):
                off = points_end + i * bytes_per_tid
                if off < target_magic_offset:
                    tids[i] = body[off]

            # 타겟(track) 정보 파싱
            track_hdr = target_magic_offset + 8
            if track_hdr + 8 <= len(body):
                _trk_frame, target_num = struct.unpack_from("<II", body, track_hdr)
                base = track_hdr + 8
                for i in range(target_num):
                    off = base + i * TARGET_SIZE
                    if off + TARGET_SIZE > len(body):
                        break
                    posX, posY = struct.unpack_from("<ff", body, off)
                    status, tid_id = struct.unpack_from("<II", body, off + 8)
                    tracks_all.append({
                        "id": int(tid_id),
                        "status": int(status),
                        "status_name": TARGET_STATUS_NAMES.get(status, "UNKNOWN"),
                        "posX": float(posX),
                        "posY": float(posY),
                    })
                if tracks_all:
                    t = tracks_all[0]
                    primary_trk = [t["id"], t["status"], t["posX"], t["posY"]]

    return {
        "T": time.time(),
        "frame_count": int(frame_count),
        "C": coords.flatten().tolist(),
        "V": velocities.tolist(),
        "P": powers.tolist(),
        "TID": tids.tolist(),
        "TRK": primary_trk,
        "_tracks_all": tracks_all,
    }


# ----- 메인 루프 -----

def main():
    print(f"📡 SRS Retina 4SN 실시간 클라이언트")
    print(f"   대상         : {RADAR_IP}:{RADAR_PORT}")
    print(f"   SAVE_JSON         : {SAVE_JSON}")
    print(f"   APPLY_PROCESSING  : {APPLY_PROCESSING}")
    print(f"   STREAM_MODEL_INPUT: {STREAM_MODEL_INPUT}\n")

    # 출력 폴더 준비
    out_dir = None
    if SAVE_JSON:
        session = SESSION_NAME or datetime.now().strftime("session_%Y%m%d_%H%M%S")
        out_dir = os.path.join(OUTPUT_BASE, session + "_frames")
        os.makedirs(out_dir, exist_ok=True)
        print(f"💾 저장 폴더 : {out_dir}")

    # cluster_process import (APPLY_PROCESSING 또는 STREAM_MODEL_INPUT 어느 하나라도 켜져 있으면 필요)
    cp = None
    if APPLY_PROCESSING or STREAM_MODEL_INPUT:
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import cluster_process as cp  # noqa
            if STREAM_MODEL_INPUT:
                print("🔧 cluster_process 로드 완료 "
                      f"(filter+voxel+SOR+DBSCAN+normalize+FPS, N={cp.MODEL_N_POINTS})\n")
            else:
                print("🔧 cluster_process 로드 완료 (V/P/Voxel/SOR/DBSCAN)\n")
        except Exception as e:
            print(f"⚠️  cluster_process 로드 실패: {e}")
            print("    전처리 없이 진행합니다.\n")
            cp = None

    total_frames = 0

    while True:
        sock = None
        try:
            # TCP 접속
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(RECV_TIMEOUT_SEC)
            print(f"⏳ 접속 시도 → {RADAR_IP}:{RADAR_PORT}")
            sock.connect((RADAR_IP, RADAR_PORT))
            print("✅ 연결 성공\n")
            sock.settimeout(None)

            fps_count = 0
            last_report = time.time()
            last_n_pts = 0
            last_tracks = []

            while True:
                frame = parse_packet(sock)
                if frame is None:
                    continue

                total_frames += 1
                fps_count += 1
                last_n_pts = len(frame["C"]) // 3
                last_tracks = frame["_tracks_all"]

                # ---- 저장 ----
                if SAVE_JSON:
                    # 확장 필드(_*) 제외하여 기존 형식과 동일하게
                    save_data = {k: v for k, v in frame.items() if not k.startswith("_")}
                    ts = f"{frame['T']:.6f}"
                    out_path = os.path.join(out_dir, f"{ts}.json")
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(save_data, f, ensure_ascii=False)

                # ---- 실시간 전처리 ----
                if cp is not None:
                    C_in = np.array(frame["C"], dtype=np.float64).reshape(-1, 3)
                    V_in = np.array(frame["V"], dtype=np.float64)
                    P_in = np.array(frame["P"], dtype=np.float64)

                    # [Mode 2] APPLY_PROCESSING — 단순 필터링 결과만
                    if APPLY_PROCESSING:
                        C2, V2, P2, cluster_id, stats = cp.process_frame(
                            C_in, V_in, P_in)
                        # 🔹 여기에 ML 추론/ROS publish/시각화 추가 가능:
                        # features = np.column_stack([C2, V2[:, None], P2[:, None]])

                    # [Mode 3] STREAM_MODEL_INPUT — 모델 입력 형태까지 변환 → 콜백
                    if STREAM_MODEL_INPUT:
                        clusters, model_stats = cp.process_frame_to_model_input(
                            C_in, V_in, P_in)
                        if clusters:
                            frame_meta = {
                                "T": frame["T"],
                                "frame_count": frame.get("frame_count", 0),
                                "n_orig_points": len(P_in),
                                "tracks": last_tracks,
                            }
                            try:
                                on_model_input(clusters, frame_meta)
                            except Exception as cb_err:
                                print(f"⚠️  on_model_input 콜백 에러: {cb_err}")

                # ---- 진단 출력 ----
                now = time.time()
                if now - last_report >= REPORT_INTERVAL_SEC:
                    fps = fps_count / (now - last_report)
                    status_list = [t["status_name"] for t in last_tracks]
                    status_str = ", ".join(status_list) if status_list else "—"
                    print(f"📊 #{total_frames:>6}  {fps:5.1f} fps  "
                          f"점 {last_n_pts:>4d}  트랙 {len(last_tracks)}  [{status_str}]")
                    fps_count = 0
                    last_report = now

        except KeyboardInterrupt:
            print(f"\n🛑 종료 요청. 총 {total_frames} frame 수신")
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
