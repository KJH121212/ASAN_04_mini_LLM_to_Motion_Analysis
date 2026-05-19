import os
import sys
import pandas as pd
from pathlib import Path

# 프로젝트 최상위 폴더 경로 설정 
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

print(f"\n📁 프로젝트 루트 경로: {PROJECT_ROOT}\n")

# Data 위치 설정
from utils.metadata.normalize_filenames import normalize_filenames
from utils.metadata.update_metadata import update_video_metadata

# --- 함수 사용 예시 (어디서든 아래처럼 호출해서 쓸 수 있습니다) ---
TARGET_DIR = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/d03"  # 탐색할 폴더 경로를 지정합니다.
INPUT_FILE = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data_d03/metadata_v1.0.csv"  # 읽어올 기존 CSV 경로를 지정합니다.
OUTPUT_FILE = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data_d03/metadata_v1.1.csv"  # 저장할 새 CSV 경로를 지정합니다.

normalize_filenames("/workspace/nas203/ds_RehabilitationMedicineData/IDs/d03")
update_video_metadata(TARGET_DIR, INPUT_FILE, OUTPUT_FILE)  # 설정한 변수들을 인자로 넣어 함수를 한 줄로 실행합니다.