"""
풀 파이프라인 전처리 (JSON 입력 → JSON 출력 / 또는 모델 입력 직접 변환):
V/P/Z 필터 + Voxel + SOR + DBSCAN + 군집 점 수 필터.

두 가지 사용 모드:

  [A] offline JSON 저장 (기존 동작)
      입력: */*_frames/ 폴더의 JSON (C, V, P, TID, TRK)
      출력: */*_preprocessed/ 폴더의 JSON (T, C, V, P [, cluster_id])
      엔트리: main() / process_frame()

  [B] 모델 입력 직접 변환 (신규)
      입력: in-memory (C, V, P) numpy 배열
      출력: per-cluster 정규화된 (N, 5) array list + 메타
      엔트리: process_frame_to_model_input(C, V, P) → List[ClusterInput]
      radar_client.py 의 STREAM_MODEL_INPUT 모드에서 콜백으로 바로 사용 가능.

처리 단계 (모드 [A], [B] 공통, 각 단계마다 C/V/P 동시 슬라이싱):
  1) V 필터    : |V| > V_THRESHOLD          (정적 반사 제거)
  2) P 필터    : P >= P_THRESHOLD           (약한 반사 제거)
  3) (옵션) Z  : Z_MIN <= z <= Z_MAX        (지면/천장 컷)
  4) Voxel    : 격자 다운샘플링 (같은 voxel의 점들은 평균)
  5) SOR      : 통계적 outlier 제거
  6) DBSCAN   : 점들을 객체 단위로 군집화
  7) 군집 필터 : 점 수 < MIN_CLUSTER_POINTS인 작은 군집 제거

모드 [B] 만의 추가 단계:
  8) per-cluster split        : DBSCAN cluster_id 로 점들을 객체별로 분리
  9) cluster local 정규화     : xy zero-mean + z floor-anchor + scale + V/P 표준화
 10) FPS 고정 길이 샘플링     : 각 cluster 점 수 → MODEL_N_POINTS 로 통일
"""

import os
import json
import glob
import numpy as np
import open3d as o3d

# ====== 설정 (필요시 여기만 수정) ======
INPUT_ROOT = "/Users/aaasa/Downloads/레이다 파일/전처리"

# --- 1) V 필터 ---
APPLY_V_FILTER = False
V_THRESHOLD = 0.0          # |V|가 이 값 초과인 점만 통과 (0이면 V=0인 정적 점 제거)

# --- 2) P 필터 ---
APPLY_P_FILTER = False
P_THRESHOLD = 5000.0       # P가 이 값 이상인 점만 통과 (반사 강도)

# --- 3) Z 게이팅 (옵션) ---
APPLY_Z_FILTER = False
Z_MIN = -2.0
Z_MAX = 2.0

# --- 4) Voxel + 5) SOR ---
VOXEL_SIZE = 0.05
SOR_NB_NEIGHBORS = 10
SOR_STD_RATIO = 3.0

# --- 6) DBSCAN ---
APPLY_DBSCAN = False
DBSCAN_EPS = 1.0           # 군집 반경 (m). 사람 크기 고려해 1m 권장
DBSCAN_MIN_POINTS = 5      # DBSCAN 내부 파라미터

# --- 7) 군집 점 수 필터 ---
MIN_CLUSTER_POINTS = 8     # 이 값 미만 군집은 버림

# 단계별 최소 점 수
MIN_POINTS_AFTER_VOXEL = 5
MIN_POINTS_AFTER_SOR = 3

# DBSCAN 군집 라벨을 출력 JSON에 포함할지 (디버깅/시각화용)
WRITE_CLUSTER_ID = False

# 폴더명
INPUT_SUFFIX = "_frames"
OUTPUT_SUFFIX = "_preprocessed"

# ====== 모드 [B] 모델 입력 변환 설정 (process_frame_to_model_input 전용) ======
# 모델 입력 형태: cluster 마다 (MODEL_N_POINTS, 5) = [x, y, z, V, P]
# - 좌표는 cluster local frame (xy centroid 기준 zero-mean, z 는 z_min 기준)
# - V, P 는 데이터셋 통계로 표준화 (V 는 mean/std, P 는 log1p 후 mean/std)
# - 점 수가 부족하면 random repetition 으로 padding, 많으면 FPS downsample
MODEL_N_POINTS = 256

# cluster local 좌표 scale (xy/z 모두 이 값으로 나눠 [-1, 1] 근처)
MODEL_NORM_SCALE = 2.0

# V 표준화 통계 (데이터셋에서 측정 후 갱신 권장)
MODEL_V_MEAN = 0.0
MODEL_V_STD  = 1.5

# P 표준화: log1p 후 통계 (long-tail 분포라 log 변환 후 표준화)
MODEL_P_LOG_MEAN = 6.0
MODEL_P_LOG_STD  = 2.0

# 모델 입력 모드에서는 DBSCAN 이 무조건 필요하므로 별도 토글로 강제 적용
# (위 APPLY_DBSCAN 토글과 무관하게 process_frame_to_model_input 안에서는 항상 ON)
# =====================================


def load_frame(path):
    """frame JSON에서 (T, C N×3, V, P) 추출. TID/TRK는 버림."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    T = float(d.get("T", 0.0))
    C = np.array(d.get("C", []), dtype=np.float64).reshape(-1, 3)
    V = np.array(d.get("V", []), dtype=np.float64)
    P = np.array(d.get("P", []), dtype=np.float64)
    n = min(len(C), len(V), len(P))
    return T, C[:n], V[:n], P[:n]


def filter_by_signal(C, V, P):
    """V/P/Z 필터. mask로 C/V/P 동시 슬라이싱."""
    mask = np.ones(len(C), dtype=bool)

    if APPLY_V_FILTER:
        mask &= np.abs(V) > V_THRESHOLD

    if APPLY_P_FILTER:
        mask &= P >= P_THRESHOLD

    if APPLY_Z_FILTER:
        z = C[:, 2]
        mask &= (z >= Z_MIN) & (z <= Z_MAX)

    return C[mask], V[mask], P[mask]


def voxel_downsample(C, V, P, voxel_size):
    """격자 다운샘플: 같은 voxel에 들어간 점들의 xyz/V/P를 평균."""
    if len(C) == 0:
        return C, V, P

    keys = np.floor(C / voxel_size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    n_voxels = int(inverse.max()) + 1

    counts = np.bincount(inverse).astype(np.float64)
    C_new = np.stack([
        np.bincount(inverse, weights=C[:, i]) / counts for i in range(3)
    ], axis=1)
    V_new = np.bincount(inverse, weights=V) / counts
    P_new = np.bincount(inverse, weights=P) / counts
    return C_new, V_new, P_new


def sor_filter(C, V, P):
    """SOR: outlier 제거 후 C/V/P 동시 슬라이싱."""
    if len(C) < MIN_POINTS_AFTER_VOXEL:
        return C, V, P

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(C)
    _, ind = pcd.remove_statistical_outlier(
        nb_neighbors=SOR_NB_NEIGHBORS, std_ratio=SOR_STD_RATIO
    )
    ind = np.asarray(ind, dtype=np.int64)
    return C[ind], V[ind], P[ind]


def dbscan_cluster_filter(C, V, P):
    """DBSCAN 후 점 수 >= MIN_CLUSTER_POINTS인 군집만 유지.
    반환: (C, V, P, labels_kept, n_clusters_kept)."""
    if len(C) < DBSCAN_MIN_POINTS:
        return C, V, P, np.full(len(C), -1, dtype=np.int64), 0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(C)
    labels = np.array(
        pcd.cluster_dbscan(
            eps=DBSCAN_EPS,
            min_points=DBSCAN_MIN_POINTS,
            print_progress=False,
        ),
        dtype=np.int64,
    )

    if len(labels) == 0:
        empty = np.empty((0,), dtype=np.int64)
        return C[:0], V[:0], P[:0], empty, 0

    keep_mask = np.zeros(len(labels), dtype=bool)
    n_clusters_kept = 0
    for label in sorted(set(labels.tolist()) - {-1}):
        cluster_mask = labels == label
        if cluster_mask.sum() >= MIN_CLUSTER_POINTS:
            keep_mask |= cluster_mask
            n_clusters_kept += 1

    return C[keep_mask], V[keep_mask], P[keep_mask], labels[keep_mask], n_clusters_kept


def process_frame(C, V, P):
    """전체 파이프라인. (C, V, P, cluster_id, stats) 반환.
    cluster_id는 DBSCAN 미적용 시 None."""
    stats = {"orig": len(C), "signal": 0, "voxel_sor": 0, "final": 0, "n_clusters": 0}
    cluster_id = None

    # 1-3) 신호 필터
    C, V, P = filter_by_signal(C, V, P)
    stats["signal"] = len(C)
    if len(C) < MIN_POINTS_AFTER_VOXEL:
        stats["final"] = len(C)
        return C, V, P, cluster_id, stats

    # 4) Voxel
    C, V, P = voxel_downsample(C, V, P, VOXEL_SIZE)
    if len(C) < MIN_POINTS_AFTER_VOXEL:
        stats["voxel_sor"] = len(C)
        stats["final"] = len(C)
        return C, V, P, cluster_id, stats

    # 5) SOR
    C, V, P = sor_filter(C, V, P)
    stats["voxel_sor"] = len(C)

    # 6-7) DBSCAN + 군집 필터
    if APPLY_DBSCAN and len(C) >= DBSCAN_MIN_POINTS:
        C, V, P, cluster_id, n_clusters = dbscan_cluster_filter(C, V, P)
        stats["n_clusters"] = n_clusters

    stats["final"] = len(C)
    return C, V, P, cluster_id, stats


def write_frame_json(out_path, T, C, V, P, cluster_id):
    """전처리 후 frame JSON 저장."""
    out = {
        "T": T,
        "C": C.flatten().tolist(),
        "V": V.tolist(),
        "P": P.tolist(),
    }
    if WRITE_CLUSTER_ID and cluster_id is not None:
        out["cluster_id"] = cluster_id.tolist()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)


def find_frame_dirs(root):
    found = []
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if d.endswith(INPUT_SUFFIX):
                found.append(os.path.join(dirpath, d))
    return sorted(found)


def main():
    frame_dirs = find_frame_dirs(INPUT_ROOT)
    print(f"📂 발견된 frame 폴더: {len(frame_dirs)}개")
    print(f"   필터: V={APPLY_V_FILTER}, P={APPLY_P_FILTER}, "
          f"Z={APPLY_Z_FILTER}, DBSCAN={APPLY_DBSCAN}")
    print(f"   임계값: V>{V_THRESHOLD}, P>={P_THRESHOLD}, "
          f"군집점수>={MIN_CLUSTER_POINTS}\n", flush=True)

    total_files = 0
    total_clusters = 0
    sums = {"orig": 0, "signal": 0, "voxel_sor": 0, "final": 0}
    failed = 0

    for frame_dir in frame_dirs:
        parent = os.path.dirname(frame_dir)
        base = os.path.basename(frame_dir)
        out_dir = os.path.join(parent, base[: -len(INPUT_SUFFIX)] + OUTPUT_SUFFIX)
        os.makedirs(out_dir, exist_ok=True)

        json_files = sorted(glob.glob(os.path.join(frame_dir, "*.json")))
        if not json_files:
            continue

        d_sums = {"orig": 0, "signal": 0, "voxel_sor": 0, "final": 0}
        d_clusters = 0

        for jf in json_files:
            try:
                T, C, V, P = load_frame(jf)
                if len(C) == 0:
                    failed += 1
                    continue

                C2, V2, P2, cluster_id, stats = process_frame(C, V, P)

                stem = os.path.splitext(os.path.basename(jf))[0]
                out_path = os.path.join(out_dir, stem + ".json")
                write_frame_json(out_path, T, C2, V2, P2, cluster_id)

                total_files += 1
                d_clusters += stats["n_clusters"]
                for k in d_sums:
                    d_sums[k] += stats[k]
            except Exception as e:
                failed += 1
                print(f"  ⚠️  {os.path.basename(jf)}: {e}")

        for k in sums:
            sums[k] += d_sums[k]
        total_clusters += d_clusters

        r = (1 - d_sums["final"] / d_sums["orig"]) * 100 if d_sums["orig"] else 0
        print(f"✅ {os.path.relpath(out_dir, INPUT_ROOT)}  "
              f"({len(json_files)} files, {d_sums['orig']:,} → {d_sums['final']:,} pts, "
              f"-{r:.1f}%, 군집 {d_clusters:,}개)", flush=True)

    print(f"\n🎉 완료: {total_files}개 파일, 총 군집 {total_clusters:,}개")
    if sums["orig"]:
        print(f"   원본:        {sums['orig']:,}")
        print(f"   ① 신호 필터: {sums['signal']:,}  "
              f"(-{(1 - sums['signal']/sums['orig'])*100:.1f}%)")
        print(f"   ② Voxel+SOR: {sums['voxel_sor']:,}  "
              f"(-{(1 - sums['voxel_sor']/sums['orig'])*100:.1f}%)")
        print(f"   ③ 군집 필터: {sums['final']:,}  "
              f"(-{(1 - sums['final']/sums['orig'])*100:.1f}%)")
    if failed:
        print(f"   실패: {failed}개")


if __name__ == "__main__":
    main()


# ====================================================================
# 모드 [B] : 모델 입력 직접 변환
# --------------------------------------------------------------------
# 기존 전처리 (V/P/Z 필터 + Voxel + SOR) + DBSCAN 까지 위 함수들을 재사용하고,
# 그 위에 per-cluster split → cluster local 정규화 → FPS 고정 길이 샘플링
# 을 추가해서 "모델이 바로 forward 에 넣을 수 있는 형태" 로 만들어 반환한다.
#
# 파일 저장 없음. radar_client.py 의 STREAM_MODEL_INPUT 모드에서 callback 으로
# 곧바로 ML inference / ROS publish / 시각화에 넘긴다.
# ====================================================================


class ClusterInput:
    """모델에 그대로 forward 할 수 있는 cluster 1 개 분량 입력.

    필드:
      points       : (MODEL_N_POINTS, 5) float32 = [x, y, z, V, P]  (정규화 후)
      cluster_id   : DBSCAN 이 부여한 정수 (frame 안에서만 유효)
      n_orig_points: FPS 샘플링 전 cluster 원본 점 수
      centroid_xy  : (2,) — 정규화 전 cluster centroid xy. 추론 후 박스 절대좌표 복원용
      z_min        : float — 정규화 전 cluster z 최소값
      scale        : float — MODEL_NORM_SCALE 와 동일 (역변환 편의용 보관)

    추론 후 모델이 출력한 박스 (Δcx, Δcy, Δcz, w, l, h, yaw) 는
    아래 식으로 절대좌표로 복원:
        cx_world = pred[0] * scale + centroid_xy[0]
        cy_world = pred[1] * scale + centroid_xy[1]
        cz_world = pred[2] * scale + z_min
        w/l/h    = pred[3:6] * scale
        yaw      = pred[6]   (각도는 변환 영향 없음)
    """

    __slots__ = ("points", "cluster_id", "n_orig_points",
                 "centroid_xy", "z_min", "scale")

    def __init__(self, points, cluster_id, n_orig_points,
                 centroid_xy, z_min, scale):
        self.points = points
        self.cluster_id = int(cluster_id)
        self.n_orig_points = int(n_orig_points)
        self.centroid_xy = centroid_xy
        self.z_min = float(z_min)
        self.scale = float(scale)

    def to_dict(self):
        """직렬화가 필요할 때 (디버그/네트워크 전송)."""
        return {
            "points":        self.points.astype(np.float32).tolist(),
            "cluster_id":    self.cluster_id,
            "n_orig_points": self.n_orig_points,
            "centroid_xy":   list(map(float, self.centroid_xy)),
            "z_min":         self.z_min,
            "scale":         self.scale,
        }


# --------------------------------------------------------------------
# 8) per-cluster split — DBSCAN 결과를 객체별로 자름
# --------------------------------------------------------------------

def _split_per_cluster(C, V, P, labels):
    """DBSCAN 결과 (C, V, P, labels) 를 cluster 단위 list 로 분리.

    labels == -1 (noise) 은 제외.
    """
    groups = []
    if labels is None or len(labels) == 0:
        return groups

    for cid in sorted(set(labels.tolist()) - {-1}):
        mask = labels == cid
        if mask.sum() == 0:
            continue
        groups.append({
            "cluster_id": int(cid),
            "C": C[mask],
            "V": V[mask],
            "P": P[mask],
        })
    return groups


# --------------------------------------------------------------------
# 9) cluster local 정규화 (좌표계 reset + scale + V/P 표준화)
# --------------------------------------------------------------------

def _normalize_cluster_points(C, V, P):
    """(N, 3) C + (N,) V + (N,) P  →  (N, 5) 정규화된 점 + transform info.

    좌표계: cluster centroid 기준 zero-mean (xy), z_min 기준 평행이동 (z).
    rotation 은 적용하지 않음 (모델이 학습으로 처리).
    """
    pts5 = np.zeros((len(C), 5), dtype=np.float32)

    # 1. xy zero-mean
    centroid_xy = C[:, :2].mean(axis=0)
    pts5[:, 0] = (C[:, 0] - centroid_xy[0]) / MODEL_NORM_SCALE
    pts5[:, 1] = (C[:, 1] - centroid_xy[1]) / MODEL_NORM_SCALE

    # 2. z floor anchor
    z_min = float(C[:, 2].min())
    pts5[:, 2] = (C[:, 2] - z_min) / MODEL_NORM_SCALE

    # 3. V 표준화
    pts5[:, 3] = (V - MODEL_V_MEAN) / max(MODEL_V_STD, 1e-6)

    # 4. P log + 표준화 (P 는 0 미만일 수 없으므로 clip 후 log1p)
    P_log = np.log1p(np.clip(P, 0.0, None))
    pts5[:, 4] = (P_log - MODEL_P_LOG_MEAN) / max(MODEL_P_LOG_STD, 1e-6)

    return pts5, centroid_xy.astype(np.float32), z_min


# --------------------------------------------------------------------
# 10) FPS 고정 길이 샘플링
# --------------------------------------------------------------------

def _fps_sample_indices(points_xyz, n_samples, rng=None):
    """Farthest Point Sampling — (N, 3+) → 인덱스 (n_samples,).

    n_samples > N 이면 random repetition 으로 padding.
    cluster_process 안에서 자체 구현 (numpy / open3d 모두 의존 추가 없음).
    """
    n = len(points_xyz)
    if n == 0:
        return np.zeros(n_samples, dtype=np.int64)
    if rng is None:
        rng = np.random

    if n <= n_samples:
        extra = rng.choice(n, size=n_samples - n, replace=True)
        return np.concatenate([np.arange(n, dtype=np.int64), extra.astype(np.int64)])

    xyz = points_xyz[:, :3]
    sampled = np.zeros(n_samples, dtype=np.int64)
    distances = np.full(n, np.inf)
    sampled[0] = rng.randint(n)
    for i in range(1, n_samples):
        last = xyz[sampled[i - 1]]
        d = np.sum((xyz - last) ** 2, axis=1)
        distances = np.minimum(distances, d)
        sampled[i] = int(np.argmax(distances))
    return sampled


# --------------------------------------------------------------------
# 모드 [B] 메인 엔트리
# --------------------------------------------------------------------

def process_frame_to_model_input(C, V, P, n_points=None, skip_prefilter=False):
    """(C, V, P) raw frame → 모델이 forward 할 수 있는 cluster list.

    ★★★ 이 함수가 모델 입력 변환의 단일 진실 소스 (Single Source of Truth) ★★★
    make_model_input.py (오프라인 배치), radar_client.py (실시간 stream) 둘 다
    이 함수를 호출. cluster_process.py 상단 상수만 바꾸면 두 곳 모두 적용됨.

    파이프라인:
        1) 신호 필터 + voxel + SOR  (입력이 이미 _preprocessed 면 skip_prefilter=True)
        2) DBSCAN 강제 적용 (APPLY_DBSCAN 토글과 무관)
        3) cluster 별 split
        4) cluster local 정규화 (xy/z/scale/V/P)
        5) FPS 로 MODEL_N_POINTS 개로 고정 길이

    Args:
        C: (N, 3) 또는 flat list. 호출하는 쪽에서 numpy 변환 안 해도 자동 처리.
        V: (N,)
        P: (N,)
        n_points: None 이면 MODEL_N_POINTS 사용. CLI / 콜백 별로 다르게 쓰고 싶을 때 override.
        skip_prefilter: True 면 1) 단계 (filter+voxel+SOR) 건너뜀.
                        - radar_client (raw 레이다 입력) → False (기본)
                        - make_model_input + _preprocessed 입력 → True (중복 방지)
                        - make_model_input + _frames 입력 → False

    Returns:
        clusters: List[ClusterInput]
                  cluster 가 하나도 없으면 빈 list.
        stats:    dict — 디버그/모니터링용
                  {"orig", "signal", "voxel_sor", "after_dbscan", "n_clusters", "final_n_points_total"}
    """
    if n_points is None:
        n_points = MODEL_N_POINTS

    # 입력 정규화 (호출 편의: list, flat list, ndarray 모두 허용)
    C = np.asarray(C, dtype=np.float64)
    if C.ndim == 1:
        C = C.reshape(-1, 3)
    V = np.asarray(V, dtype=np.float64).reshape(-1)
    P = np.asarray(P, dtype=np.float64).reshape(-1)
    n = min(len(C), len(V), len(P))
    C, V, P = C[:n], V[:n], P[:n]

    stats = {
        "orig": len(C),
        "signal": 0,
        "voxel_sor": 0,
        "after_dbscan": 0,
        "n_clusters": 0,
        "final_n_points_total": 0,
    }

    if len(C) == 0:
        return [], stats

    # 1) 신호 + voxel + SOR — skip_prefilter=True 면 통째로 건너뜀
    if not skip_prefilter:
        C, V, P = filter_by_signal(C, V, P)
        stats["signal"] = len(C)
        if len(C) < MIN_POINTS_AFTER_VOXEL:
            return [], stats

        C, V, P = voxel_downsample(C, V, P, VOXEL_SIZE)
        if len(C) < MIN_POINTS_AFTER_VOXEL:
            stats["voxel_sor"] = len(C)
            return [], stats

        C, V, P = sor_filter(C, V, P)
        stats["voxel_sor"] = len(C)
    else:
        # 이미 _preprocessed 입력 — 현재 점 수 그대로 통계만 기록
        stats["signal"] = len(C)
        stats["voxel_sor"] = len(C)

    if len(C) < DBSCAN_MIN_POINTS:
        return [], stats

    # 4) DBSCAN 강제 적용 (모델 입력 모드에서는 항상 ON)
    #    dbscan_cluster_filter 는 MIN_CLUSTER_POINTS 미만 군집까지 자동 제거.
    C, V, P, labels, n_clusters = dbscan_cluster_filter(C, V, P)
    stats["after_dbscan"] = len(C)
    stats["n_clusters"] = n_clusters

    if n_clusters == 0:
        return [], stats

    # 5) per-cluster split
    groups = _split_per_cluster(C, V, P, labels)

    # 6) 각 cluster 별 정규화 + FPS sampling
    rng = np.random.RandomState()
    out = []
    for g in groups:
        gC, gV, gP = g["C"], g["V"], g["P"]
        pts5, centroid_xy, z_min = _normalize_cluster_points(gC, gV, gP)

        sel = _fps_sample_indices(pts5, n_points, rng=rng)
        sampled = pts5[sel]

        out.append(ClusterInput(
            points=sampled.astype(np.float32),
            cluster_id=g["cluster_id"],
            n_orig_points=len(gC),
            centroid_xy=centroid_xy,
            z_min=z_min,
            scale=MODEL_NORM_SCALE,
        ))
        stats["final_n_points_total"] += n_points

    return out, stats


# --------------------------------------------------------------------
# 배치 처리 헬퍼 — 여러 ClusterInput 을 (B, N, 5) numpy / torch 텐서로 stack
# --------------------------------------------------------------------

def stack_model_input(clusters):
    """List[ClusterInput] → 모델 forward 용 (B, N, 5) numpy 배열 + meta list.

    호출하는 쪽에서 torch.from_numpy().to(device) 만 추가하면 forward 가능.
    """
    if not clusters:
        return (np.zeros((0, MODEL_N_POINTS, 5), dtype=np.float32),
                [])
    points = np.stack([c.points for c in clusters], axis=0)
    meta = [{
        "cluster_id":    c.cluster_id,
        "n_orig_points": c.n_orig_points,
        "centroid_xy":   c.centroid_xy.tolist(),
        "z_min":         c.z_min,
        "scale":         c.scale,
    } for c in clusters]
    return points, meta
