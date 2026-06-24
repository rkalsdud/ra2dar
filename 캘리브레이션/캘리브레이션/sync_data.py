import os
import glob
import shutil

# ====== 경로 설정 (서버 환경에 맞게 수정하세요) ======
# 라즈베리파이에서 가져온 폴더 경로
RADAR_DIR = "./recorded_data/session_20260501_123000_frames" 
IMAGES_DIR = "./recorded_data/session_20260501_123000_images"

# 매칭된 짝꿍 데이터가 저장될 새로운 폴더
OUTPUT_DIR = "./matched_dataset"

# 허용할 최대 시간 오차 (초 단위, 0.05 = 50ms)
TIME_THRESHOLD = 0.05 
# ===================================================

def get_timestamp(filepath):
    """파일명에서 타임스탬프 숫자를 추출합니다."""
    basename = os.path.basename(filepath)
    ts_str = os.path.splitext(basename)[0]
    return float(ts_str)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    radar_files = sorted(glob.glob(os.path.join(RADAR_DIR, "*.json")))
    image_files = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))

    if not radar_files or not image_files:
        print("데이터를 찾을 수 없습니다. 경로를 확인하세요.")
        return

    # 비교 속도를 높이기 위해 이미지 시간을 미리 리스트로 생성
    print("데이터 로딩 중...")
    image_times = [get_timestamp(f) for f in image_files]
    
    matched_count = 0
    print(f"동기화 매칭 시작 (허용 오차: {TIME_THRESHOLD}초 이내)")

    for radar_file in radar_files:
        radar_ts = get_timestamp(radar_file)

        # 레이다 시간과 가장 오차가 적은 이미지 찾기
        differences = [abs(radar_ts - img_ts) for img_ts in image_times]
        min_idx = differences.index(min(differences))
        min_diff = differences[min_idx]

        # 오차가 기준치 이내라면 세트로 묶어서 복사
        if min_diff <= TIME_THRESHOLD:
            matched_img = image_files[min_idx]
            
            # 새 파일명은 레이다 타임스탬프 기준으로 통일
            base_name = f"{radar_ts:.6f}"
            
            shutil.copy(radar_file, os.path.join(OUTPUT_DIR, f"{base_name}.json"))
            shutil.copy(matched_img, os.path.join(OUTPUT_DIR, f"{base_name}.jpg"))
            
            matched_count += 1

    print(f"\n✅ 작업 완료!")
    print(f"총 {len(radar_files)}개의 레이다 데이터 중, {matched_count}쌍이 성공적으로 매칭되었습니다.")
    print(f"결과 확인: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()