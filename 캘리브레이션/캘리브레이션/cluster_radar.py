"""
레이더 JSON 파일을 객체별로 분류하는 스크립트

사용법:
    python cluster_radar.py <json_file_path>

출력 형식:
    객체1: [[x,y,z], [x,y,z], ...]
    객체2: [[x,y,z], [x,y,z], ...]
    ...
"""
import json
import sys
import numpy as np


def load_radar_json(path):
    """JSON 파일에서 C(좌표), V, P, TID 추출"""
    with open(path, 'r') as f:
        data = json.load(f)
    points = np.array(data['C']).reshape(-1, 3)
    info = {
        'V':   np.array(data.get('V',   [])),
        'P':   np.array(data.get('P',   [])),
        'TID': np.array(data.get('TID', [])),
    }
    return points, info


def dbscan_cluster(points, eps=0.35, min_samples=8):
    """
    간단한 DBSCAN 구현 (외부 라이브러리 없이).
    - eps: 같은 객체로 묶일 최대 거리 (m). 강아지/사람 같은 객체 크기를 고려해 0.3~0.5 권장
    - min_samples: 클러스터로 인정할 최소 포인트 수

    반환: 각 점의 클러스터 라벨 배열 (-1 = noise)
    """
    n = len(points)
    labels = -np.ones(n, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    # 거리 행렬 미리 계산 (점 개수 적당하면 가능)
    # 점이 많으면 KDTree를 쓰는 게 좋지만, 1만개 이하면 충분
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        # i의 이웃 찾기
        d = np.linalg.norm(points - points[i], axis=1)
        neighbors = np.where(d <= eps)[0].tolist()
        if len(neighbors) < min_samples:
            continue   # noise (다음에 다른 점의 이웃이 되면 클러스터에 흡수 가능)
        # 새 클러스터 시작
        labels[i] = cluster_id
        k = 0
        while k < len(neighbors):
            j = neighbors[k]
            k += 1
            if not visited[j]:
                visited[j] = True
                dj = np.linalg.norm(points - points[j], axis=1)
                nj = np.where(dj <= eps)[0].tolist()
                if len(nj) >= min_samples:
                    for x in nj:
                        if x not in neighbors:
                            neighbors.append(x)
            if labels[j] == -1:
                labels[j] = cluster_id
        cluster_id += 1
    return labels


def cluster_radar_points(json_path, eps=0.35, min_samples=8,
                         filter_floor=False, floor_z=None):
    """
    JSON을 읽어 객체별로 점을 분류하고 dict로 반환.

    Parameters
    ----------
    eps           : 같은 객체로 묶일 거리 임계값 (m)
    min_samples   : 클러스터 최소 점 개수
    filter_floor  : True면 바닥 가까운 점 제외
    floor_z       : 바닥으로 간주할 Z값 (None이면 자동: 전체의 5% 분위)
    """
    points, info = load_radar_json(json_path)
    keep_idx = np.arange(len(points))

    if filter_floor:
        if floor_z is None:
            floor_z = np.percentile(points[:, 2], 5) + 0.05
        keep_mask = points[:, 2] > floor_z
        points = points[keep_mask]
        keep_idx = keep_idx[keep_mask]

    labels = dbscan_cluster(points, eps=eps, min_samples=min_samples)

    # 라벨별로 그룹화
    objects = {}
    for lbl in sorted(set(labels)):
        if lbl == -1:
            continue
        mask = labels == lbl
        cluster_points = points[mask]
        cluster_orig_idx = keep_idx[mask]
        centroid = cluster_points.mean(axis=0)
        objects[int(lbl)] = {
            'points':      cluster_points.tolist(),
            'count':       int(mask.sum()),
            'centroid':    centroid.tolist(),
            'x_range':     [float(cluster_points[:, 0].min()),
                            float(cluster_points[:, 0].max())],
            'y_range':     [float(cluster_points[:, 1].min()),
                            float(cluster_points[:, 1].max())],
            'z_range':     [float(cluster_points[:, 2].min()),
                            float(cluster_points[:, 2].max())],
            'orig_indices': cluster_orig_idx.tolist(),
        }
    return objects


def print_summary(objects):
    """객체별 요약 출력"""
    print(f"\n=== 검출된 객체 수: {len(objects)} ===\n")
    # centroid Y(거리) 순으로 정렬해 출력
    sorted_ids = sorted(objects.keys(), key=lambda k: objects[k]['centroid'][1])
    for new_id, lbl in enumerate(sorted_ids, start=1):
        obj = objects[lbl]
        cx, cy, cz = obj['centroid']
        print(f"객체 {new_id} (cluster_id={lbl}):")
        print(f"  포인트 수: {obj['count']}")
        print(f"  중심  : [{cx:+.4f}, {cy:+.4f}, {cz:+.4f}]")
        print(f"  X 범위: [{obj['x_range'][0]:+.3f}, {obj['x_range'][1]:+.3f}]")
        print(f"  Y 범위: [{obj['y_range'][0]:+.3f}, {obj['y_range'][1]:+.3f}]")
        print(f"  Z 범위: [{obj['z_range'][0]:+.3f}, {obj['z_range'][1]:+.3f}]")
        # 미리보기 점 5개
        print(f"  포인트 미리보기 (앞 3개):")
        for p in obj['points'][:3]:
            print(f"    [{p[0]:+.6f}, {p[1]:+.6f}, {p[2]:+.6f}]")
        print()


def to_simple_list(objects):
    """간단한 list-of-list 형태로 반환: [[[x,y,z],...], [[x,y,z],...], ...]"""
    return [obj['points'] for obj in objects.values()]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python cluster_radar.py <json_file>")
        print("옵션: --eps 0.35 --min 8 --no-floor")
        sys.exit(1)

    json_path = sys.argv[1]
    eps = 0.35
    min_samples = 8
    filter_floor = False

    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == '--eps':
            eps = float(args[i+1]); i += 2
        elif args[i] == '--min':
            min_samples = int(args[i+1]); i += 2
        elif args[i] == '--no-floor':
            filter_floor = False; i += 1
        elif args[i] == '--floor':
            filter_floor = True; i += 1
        else:
            i += 1

    print(f"파일: {json_path}")
    print(f"파라미터: eps={eps}, min_samples={min_samples}, filter_floor={filter_floor}")
    objs = cluster_radar_points(json_path, eps=eps, min_samples=min_samples,
                                filter_floor=filter_floor)
    print_summary(objs)

    # 결과를 JSON 파일로도 저장 (현재 작업 디렉토리에)
    import os
    base = os.path.basename(json_path).replace('.json', '_clustered.json')
    out_path = os.path.join(os.getcwd(), base)
    # JSON dict 키는 문자열로 변환 (numpy int 호환 + Python 3.14 호환)
    objs_for_json = {str(k): v for k, v in objs.items()}
    with open(out_path, 'w') as f:
        json.dump({
            'object_count': len(objs_for_json),
            'objects': objs_for_json,
        }, f, indent=2)
    print(f"💾 클러스터링 결과 저장: {out_path}")