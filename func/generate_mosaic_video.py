import os
import sys
import cv2  # 🌟 FPS 추출을 위해 추가
import pandas as pd
from pathlib import Path
from tabulate import tabulate

# =======================================================
# 프로젝트 최상위 폴더 경로 설정 
# =======================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

print(f"📁 프로젝트 루트 경로: {PROJECT_ROOT}")

# =======================================================
# Data 위치 설정
# =======================================================
from utils.path_list_d03 import path_list_d03

DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data_d03")
CSV_PATH = DATA_DIR / "metadata_v1.0.csv"

df = pd.read_csv(str(CSV_PATH))

target = 0
common_path = df.iloc[target]['common_path']
paths = path_list_d03(common_path)

# =======================================================
# 🌟 [추가됨] 원본 비디오의 실제 FPS 추출
# =======================================================
source_video_path = paths['video']
cap = cv2.VideoCapture(str(source_video_path))
if not cap.isOpened():
    print(f"❌ [에러] 비디오를 열 수 없습니다: {source_video_path}")
    sys.exit()
video_fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
print(f"🎥 원본 영상 FPS 감지됨: {video_fps}")

# =======================================================
# 파싱 (시간 -> 프레임 인덱스 변환 적용)
# =======================================================
from utils.ass_parser import read_ass_subtitles, parse_bayley_subtitle_data

extracted_list = read_ass_subtitles(paths['ass'])
# 🌟 파싱할 때 video_fps를 넘겨주어 반환값을 시간(String)에서 프레임(Int)으로 변환합니다.
raw_final_data = parse_bayley_subtitle_data(extracted_list, fps=video_fps)

# =======================================================
# Video Generation
# =======================================================
from utils.video_processor.generate_skeleton_video import generate_integrated_video

print("\n🧪 첫 번째 구간(Clip)에 대해서만 테스트 렌더링을 진행합니다...")

if raw_final_data:
    # 테스트를 위해 첫 번째 자막 데이터만 가져옵니다.
    # 구조: [start_idx, end_idx, domain, item_num, score]
    first_clip = raw_final_data[0]
    
    start_idx = first_clip[0]  # 정수형 프레임 번호 (예: 812)
    end_idx = first_clip[1]    # 정수형 프레임 번호 (예: 1450)
    domain = first_clip[2]
    item_num = first_clip[3]

    output_filename = f"../data/test_{domain}_{item_num}.mp4"

    generate_integrated_video(
        frame_dir=paths['frame'],
        output_path=output_filename,
        skeleton_dir=paths['keypoint'],
        start_idx=start_idx,          # 변환된 프레임 시작점
        end_idx=end_idx,              # 변환된 프레임 종료점
        conf_threshold=0.0,
        fps=int(video_fps),           # 추출한 FPS 적용
        apply_mosaic=True,            # 모자이크 켜기 (선택 사항)
        draw_skeleton=True            # 뼈대 그리기 켜기 (선택 사항)
    )
    
    print(f"✅ 테스트 비디오 저장 완료: {output_filename}")
else:
    print("⚠️ 파싱된 자막 데이터가 없습니다.")