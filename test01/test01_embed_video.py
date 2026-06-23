# ==============================================================================
# 🚀 1단계: 비디오 프레임 특징 추출 및 Vector DB(FAISS) 구축
# - 전처리 없는 통 이미지 프레임을 순수 ViT(Vision Transformer) 백본 모델에 입력
# - 추출된 768차원의 밀집 특징 벡터를 로컬(.npy) 및 FAISS(.faiss) 데이터베이스에 캐싱
# ==============================================================================

import os
import sys
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTModel
import faiss
import umap
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 1. 프로젝트 최상위 폴더 및 경로 설정 
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
print(f"📁 프로젝트 루트 경로: {PROJECT_ROOT}")

# 💾 저장할 데이터 디렉토리 설정 (프로젝트 루트 하위의 data 폴더)
SAVE_DIR = PROJECT_ROOT / "test01"
SAVE_DIR.mkdir(parents=True, exist_ok=True) # 폴더가 없으면 자동 생성

FAISS_PATH = SAVE_DIR / "bayley_gross_motor.faiss"
FEATURES_PATH = SAVE_DIR / "bayley_features.npy"

# Data 위치 설정
from utils.path_list_d03 import path_list_d03

DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data_d03")
CSV_PATH = DATA_DIR / "metadata_v1.0.csv"

df = pd.read_csv(str(CSV_PATH))

target = 0
common_path = df.iloc[target]['common_path']
paths = path_list_d03(common_path)

# 프레임 폴더 지정 및 불러오기
FRAME_DIR = Path(paths['frame'])
print(f"🖼️ 프레임 데이터 위치: {FRAME_DIR}")

frame_paths = sorted(list(FRAME_DIR.rglob("*.jpg")))
print(f"📊 총 추출된 프레임 수: {len(frame_paths)}장")

if len(frame_paths) == 0:
    raise FileNotFoundError("지정된 경로에 .jpg 파일이 없습니다. 경로를 다시 확인해주세요.")

# 2. 캐시된 파일이 있는지 확인 (이미 뽑아놓은 벡터가 있다면 로드하고 통과)
if FAISS_PATH.exists() and FEATURES_PATH.exists():
    print("\n♻️ 이미 추출된 로컬 Vector DB 및 피처 파일이 존재합니다! 캐시에서 직접 불러옵니다.")
    features = np.load(str(FEATURES_PATH))
    index = faiss.read_index(str(FAISS_PATH))
    print(f"📦 Vector DB 복원 완료! (등록된 벡터 수: {index.ntotal}개, 피처 크기: {features.shape})")

else:
    # 3. 모델 및 프로세서 로드 (Vision Transformer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 사용 중인 디바이스: {device}")

    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224').to(device)
    model.eval()

    # 4. 프레임별 임베딩 추출 (Feature Extraction)
    features = []

    print("🚀 프레임 임베딩 추출 중...")
    with torch.no_grad():
        for f_path in tqdm(frame_paths):
            image = Image.open(f_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
            features.append(cls_embedding)

    features = np.array(features).astype('float32')
    print(f"✨ 임베딩 완료! 행렬 크기: {features.shape}")

    # 5. FAISS (Vector DB) 구축 및 인덱싱
    dimension = features.shape[1]
    faiss.normalize_L2(features)
    index = faiss.IndexFlatIP(dimension)
    index.add(features)
    print("📦 FAISS Vector DB 구축 완료.")

    # 💾 로컬 하드디스크(../data)에 저장하기
    np.save(str(FEATURES_PATH), features)
    faiss.write_index(index, str(FAISS_PATH))
    print(f"💾 피처 데이터 및 Vector DB 가 로컬에 저장되었습니다:\n  - Vector DB: {FAISS_PATH}\n  - Features: {FEATURES_PATH}")


# 6. 시간 축 변위 분석 (Divider 기능)
distances = []
for i in range(len(features) - 1):
    sim = np.dot(features[i], features[i+1])
    distances.append(1 - sim)

threshold = np.percentile(distances, 99)
divider_frames = [i for i, d in enumerate(distances) if d > threshold]
print(f"🚨 동작이 급변하는 Divider 후보 프레임 (상위 1%): {divider_frames[:10]} ...")


# 7. UMAP 차원 축소 및 시간 흐름 시각화
print("🎨 Latent Space 시각화를 위한 차원 축소 중 (UMAP)...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
embedding_2d = reducer.fit_transform(features)

plt.figure(figsize=(12, 8))
sc = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                 c=range(len(embedding_2d)), cmap='viridis', s=15, alpha=0.6, label='Frames')

plt.plot(embedding_2d[:, 0], embedding_2d[:, 1], color='gray', linewidth=0.5, alpha=0.4)

plt.scatter(embedding_2d[0, 0], embedding_2d[0, 1], color='red', s=100, marker='*', label='START', zorder=5)
plt.scatter(embedding_2d[-1, 0], embedding_2d[-1, 1], color='blue', s=100, marker='X', label='END', zorder=5)

plt.colorbar(sc, label="Frame Timeline (Index)")
plt.title("🔑 Frame-level Latent Space Trajectory (ViT + UMAP)", fontsize=14)
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

output_plot = "./frame_trajectory_result.png"
plt.savefig(str(output_plot), dpi=300, bbox_inches='tight')
print(f"💾 시각화 결과 이미지 저장 완료: {output_plot}")
plt.show()