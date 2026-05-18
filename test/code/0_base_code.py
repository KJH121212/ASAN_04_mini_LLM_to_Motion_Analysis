import os
import sys
import pandas as pd
from pathlib import Path

# 프로젝트 최상위 폴더 경로 설정 
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

print(f"📁 프로젝트 루트 경로: {PROJECT_ROOT}")

# Data 위치 설정
from utils.path_list_d03 import path_list_d03

DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data_d03")  # 데이터 디렉토리 경로를 설정합니다.
CSV_PATH = DATA_DIR / "metadata_v1.0.csv"

df = pd.read_csv(str(CSV_PATH))

target = 0

common_path = df.iloc[target]['common_path']
paths = path_list_d03(common_path)

paths['ass']