import cv2
import numpy as np
import json
import os
import glob

# ------ 1. 고정 파라미터 (이전 단계에서 구한 값) ------
camera_matrix = np.array([
    [800.0, 0, 640.0],
    [0, 800.0, 360.0],
    [0, 0, 1]
], dtype=np.float32)
#dist_coeffs = np.zeros((4, 1))
dist_coeffs = np.array([-0.40, 0.15, 0.0, 0.0, 0.0], dtype=np.float32) #인터넷에 나온 왜곡 보정 값

# rvec = np.array([[1.21168672], [0.81984146], [1.05531676]], dtype=np.float32)
# tvec = np.array([[-2.02654158], [-4.18238423], [8.41313908]], dtype=np.float32)
rvec = np.array([[-4.80094906],
 [-0.68653157],
 [ 0.54689392]], dtype=np.float32)
tvec = np.array([[-0.07623432],
 [-1.42401822],
 [ 0.44725201]], dtype=np.float32)



# ======================================
# 회전 미세 조정 (도 단위)
# ======================================
rot_x = -3     # 위/아래 기울기
rot_y = -27    # 바닥 기준 좌우 회전
rot_z = 3      # 화면 회전

rx = np.deg2rad(rot_x)
ry = np.deg2rad(rot_y)
rz = np.deg2rad(rot_z)

# rvec → 회전행렬
R, _ = cv2.Rodrigues(rvec)

# X축 회전
Rx = np.array([
    [1, 0, 0],
    [0, np.cos(rx), -np.sin(rx)],
    [0, np.sin(rx), np.cos(rx)]
], dtype=np.float32)

# Y축 회전
Ry = np.array([
    [np.cos(ry), 0, np.sin(ry)],
    [0, 1, 0],
    [-np.sin(ry), 0, np.cos(ry)]
], dtype=np.float32)

# Z축 회전
Rz = np.array([
    [np.cos(rz), -np.sin(rz), 0],
    [np.sin(rz), np.cos(rz), 0],
    [0, 0, 1]
], dtype=np.float32)

# 추가 회전 적용
R_new = Rz @ Ry @ Rx @ R

# 다시 rvec 변환
rvec, _ = cv2.Rodrigues(R_new)

print("보정 후 rvec:")
print(rvec)
# ======================================




# ------ 2. 경로 설정 ------
INPUT_DIR = "./matched_dataset"  # 동기화된 데이터 폴더
IMAGE_OUT_DIR = "./batch_results" # 시각화 이미지 저장 폴더
LABEL_OUT_DIR = "./projected_labels" # 좌표 데이터 저장 폴더

os.makedirs(IMAGE_OUT_DIR, exist_ok=True)
os.makedirs(LABEL_OUT_DIR, exist_ok=True)

def main():
    json_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))
    print(f"총 {len(json_files)}개의 프레임 통합 처리를 시작합니다...")

    for json_path in json_files:
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        
        # 이미지 파일 찾기 (png, jpg 대응)
        img_path = os.path.join(INPUT_DIR, f"{base_name}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(INPUT_DIR, f"{base_name}.jpg")
        
        if not os.path.exists(img_path):
            continue

        # 데이터 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            radar_data = json.load(f)
        
        img = cv2.imread(img_path)
        if img is None: continue

        coords = radar_data.get("C", [])
        powers = radar_data.get("P", [])
        velocities = radar_data.get("V", [])
        tids = radar_data.get("TID", [])

        if len(coords) < 3: continue

        # 3D 좌표를 2D로 투영
        points_3d = np.array(coords, dtype=np.float32).reshape(-1, 3)
        points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs)

        projected_list = []
        
        # 점 그리기와 데이터 수집을 동시에 수행
        for i, p in enumerate(points_2d):
            u, v = p[0][0], p[0][1]
            
            # 유효 범위(1280x720) 확인
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                # 1) 시각화: 사진 위에 점 그리기
                cv2.circle(img, (int(u), int(v)), 5, (0, 0, 255), -1)
                cv2.circle(img, (int(u), int(v)), 6, (0, 255, 0), 1)

                # 2) 데이터 추출: JSON 저장을 위한 리스트 추가
                projected_list.append({
                    "radar_idx": i,
                    "pixel_x": float(u),
                    "pixel_y": float(v),
                    "radar_3d": points_3d[i].tolist(),
                    "power": float(powers[i]) if i < len(powers) else None,
                    "velocity": float(velocities[i]) if i < len(velocities) else None
                    "radar_TID": int(tids[i]) if i < len(tids) else 0
                })

        # 결과물 저장
        cv2.imwrite(os.path.join(IMAGE_OUT_DIR, f"res_{base_name}.jpg"), img)
        with open(os.path.join(LABEL_OUT_DIR, f"{base_name}_projected.json"), "w", encoding="utf-8") as f:
            json.dump(projected_list, f, indent=4)

    print(f"✅ 통합 처리 완료!")
    print(f"이미지 결과: {IMAGE_OUT_DIR}")
    print(f"데이터 결과: {LABEL_OUT_DIR}")

if __name__ == "__main__":
    main()