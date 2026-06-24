import os
import time
from datetime import datetime
import cv2
from picamera2 import Picamera2

# ====== 설정 ======
OUTPUT_BASE = "./recorded_data"
SESSION_NAME = None  
WIDTH = 1280
HEIGHT = 720
TARGET_FPS = 21
# ==================

def main():
    print("📷 RPi Camera v3 전용 수집기 (Picamera2 버전) 시작")
    
    session = SESSION_NAME or datetime.now().strftime("session_%Y%m%d_%H%M%S")
    images_dir = os.path.join(OUTPUT_BASE, session + "_images")
    os.makedirs(images_dir, exist_ok=True)
    print(f"💾 저장 폴더: {images_dir}\n")

    # 1. 라즈베리파이 공식 카메라 라이브러리 초기화
    try:
        picam2 = Picamera2()
    except Exception as e:
        print(f"⚠️ 카메라 초기화 실패: {e}")
        print("케이블이 잘 꽂혀있는지, 또는 다른 프로그램이 카메라를 쓰고 있지 않은지 확인하세요.")
        return
    
    # 2. 해상도 및 포맷 설정
    config = picam2.create_video_configuration(main={"size": (WIDTH, HEIGHT)})
    picam2.configure(config)
    picam2.start()

    print("✅ 수집 시작 — Ctrl+C로 종료\n")
    total_frames = 0

    #fps추가
    frame_duration = 1.0 / TARGET_FPS
    #
    try:
        while True:
            #fps추가
            loop_start = time.time()
            #
            # 3. 최신 프레임을 가져오고 즉시 타임스탬프 기록
            frame = picam2.capture_array("main")
            capture_time = time.time()
            
            # 4. 파일명 생성 및 색상 변환 후 저장 (Picamera2는 RGB, OpenCV는 BGR을 쓰기 때문)
            ts_str = f"{capture_time:.6f}"
            img_path = os.path.join(images_dir, f"{ts_str}.jpg")
            
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path, frame_bgr)
            
            total_frames += 1
            if total_frames % 30 == 0:
                print(f"상태: {total_frames}장 저장 완료 (최근: {ts_str}.jpg)")


            #fps추가
            process_time = time.time() - loop_start # 사진 찍고 저장하는데 걸린 시간 측정
            sleep_time = frame_duration - process_time # 21fps를 맞추기 위해 남은 시간 계산
            if sleep_time > 0:
                time.sleep(sleep_time)
            #

    except KeyboardInterrupt:
        print(f"\n🛑 수집 종료. 총 {total_frames}장 저장 완료")
    finally:
        picam2.stop() # 안전한 종료

if __name__ == "__main__":
    main()