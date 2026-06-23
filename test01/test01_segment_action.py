# ==============================================================================
# 🎬 2단계: 비지도 학습 기반 동작 클러스터링 및 비디오 자동 분할 (통합 완결판)
# - test01의 768차원 풀 프레임 특징 벡터를 DBSCAN 알고리즘으로 군집화
# - UMAP 차원 축소 후, 전문의가 라벨링한 .ass 자막 데이터(Ground Truth)를 매핑
# - 시간 연속성(30프레임 이상)을 감지하여 독립된 원본 비디오 클립으로 자동 저장
# ==============================================================================

import os # 디렉토리 생성 및 시스템 관리를 위해 기본 모듈을 불러옵니다.
import sys # 시스템 패스 제어 및 경로 추가를 위해 불러옵니다.
import json # JSON 파일 확인용으로 불러옵니다.
import re # 자막 안의 타임스탬프 문자를 추출하기 위해 정규표현식 모듈을 불러옵니다.
import pandas as pd # 메타데이터 CSV 파일 처리를 위해 불러옵니다.
import numpy as np # 행렬 계산 및 배열 로드를 위해 불러옵니다.
import cv2 # 비디오 클립(.mp4) 파일 생성을 위해 OpenCV를 불러옵니다.
from pathlib import Path # 경로 처리를 객체지향적으로 다루기 위해 불러옵니다.
from datetime import datetime, timedelta # 시간 포맷을 파싱하고 프레임으로 변환하기 위해 불러옵니다.
from sklearn.cluster import DBSCAN # 군집 개수를 스스로 찾는 밀도 기반 DBSCAN 알고리즘입니다.
import umap # 768차원 벡터를 2차원 공간으로 축소하기 위해 불러옵니다.
import matplotlib.pyplot as plt # UMAP 그래프 시각화 및 저장을 위해 불러옵니다.
from collections import Counter # 군집별 프레임 분포 개수를 쉽게 확인하기 위해 불러옵니다.
from tqdm import tqdm # 비디오 인코딩 등 반복문 진행률 표시를 위해 불러옵니다.

# ==============================================================================
# 🛠️ 함수 정의 부
# ==============================================================================
def timestamp_to_frame_idx(ts_str, fps=30.0):
    """
    '0:00:23.90' 형태의 자막 타임스탬프를 프레임 인덱스(정수)로 변환합니다.
    """
    try:
        t = datetime.strptime(ts_str.strip(), "%H:%M:%S.%f")
    except ValueError:
        t = datetime.strptime(ts_str.strip(), "%M:%S.%f")
        
    delta = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond)
    return int(delta.total_seconds() * fps)


# ==============================================================================
# 📂 1. 경로 및 데이터 설정
# ==============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
print(f"📁 프로젝트 루트 경로: {PROJECT_ROOT}")

SAVE_DIR = PROJECT_ROOT / "test01" 
FEATURES_PATH = SAVE_DIR / "bayley_features.npy"
CLIPS_OUT_DIR = SAVE_DIR / "clips"
CLIPS_OUT_DIR.mkdir(parents=True, exist_ok=True)

if not FEATURES_PATH.exists():
    raise FileNotFoundError(f"❌ {FEATURES_PATH} 파일이 없습니다. 1단계 코드를 먼저 실행해 주세요.")

from utils.path_list_d03 import path_list_d03 
DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data_d03") 
CSV_PATH = DATA_DIR / "metadata_v1.0.csv" 
df = pd.read_csv(str(CSV_PATH)) 

target = 0 
common_path = df.iloc[target]['common_path'] 
paths = path_list_d03(common_path) 
FRAME_DIR = Path(paths['frame']) 


# ==============================================================================
# 🧩 2. 피처 로드 및 전체 프레임 다이렉트 매핑
# ==============================================================================
features = np.load(str(FEATURES_PATH)) 
print(f"📦 캐시된 피처 로드 완료. 크기: {features.shape}") 

frame_paths = sorted(list(FRAME_DIR.rglob("*.jpg"))) 
print(f"🔍 디렉토리 내 실제 이미지 수: {len(frame_paths)}장") 

# 안전 예외 처리 및 싱크 정렬
valid_frame_paths = frame_paths.copy()
if len(valid_frame_paths) != features.shape[0]: 
    print("⚠️ 경고: 이미지 수와 피처 라벨 수가 일치하지 않습니다. 피처 개수에 맞춰 프레임 리스트를 조정합니다.")
    valid_frame_paths = valid_frame_paths[:features.shape[0]]


# ==============================================================================
# 🤖 3. DBSCAN 클러스터링 수행 (자동 군집화)
# ==============================================================================
print("\n🤖 DBSCAN 알고리즘으로 밀도 기반 클러스터링을 진행합니다...") 
dbscan = DBSCAN(eps=0.3, min_samples=30, metric='euclidean') 
cluster_labels = dbscan.fit_predict(features) 

unique_labels = set(cluster_labels) 
num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0) 

print(f"✅ AI가 스스로 찾아낸 동작 군집 개수: {num_clusters}개") 
print(f"📊 군집별 프레임 분포 (-1은 노이즈): {dict(Counter(cluster_labels))}") 

# ==============================================================================
# 🎨 4. 다각적 시각화를 위한 차원 축소 수행 (UMAP, t-SNE, PCA)
# ==============================================================================
print("\n🎨 잠재 공간 다각적 분석을 위한 차원 축소 시작 (3종 세트)...") 

# ① UMAP 축소 (Global/Local 균형)
print("⏳ 1/3 UMAP 변환 중...")
reducer_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42) 
embedding_umap = reducer_umap.fit_transform(features) 

# ② t-SNE 축소 (Local 구조 극대화, 덩어리 분리 강조)
print("⏳ 2/3 t-SNE 변환 중...")
from sklearn.manifold import TSNE
# 768차원 이미지 벡터인 경우 코사인 거리가 유리하므로 metric='cosine' 적용
reducer_tsne = TSNE(n_components=2, perplexity=30, metric='cosine', init='random', random_state=42, n_jobs=-1)
embedding_tsne = reducer_tsne.fit_transform(features)

# ③ PCA 축소 (선형 분산 극대화, 기하학적 왜곡 없음)
print("⏳ 3/3 PCA 변환 중...")
from sklearn.decomposition import PCA
reducer_pca = PCA(n_components=2, random_state=42)
embedding_pca = reducer_pca.fit_transform(features)


# ==============================================================================
# ✂️ 5. 시간 흐름에 따른 연속 구간(Segments) 추출 알고리즘 (기존 유지)
# ==============================================================================
MIN_SEGMENT_LENGTH = 30 
segments = [] 
start_idx = 0 
current_label = cluster_labels[0] 

for i in range(1, len(cluster_labels)): 
    if cluster_labels[i] != current_label: 
        end_idx = i - 1 
        length = end_idx - start_idx + 1 
        if length >= MIN_SEGMENT_LENGTH: 
            segments.append({'label': current_label, 'start': start_idx, 'end': end_idx, 'length': length}) 
        start_idx = i 
        current_label = cluster_labels[i] 

end_idx = len(cluster_labels) - 1 
if (end_idx - start_idx + 1) >= MIN_SEGMENT_LENGTH: 
    segments.append({'label': current_label, 'start': start_idx, 'end': end_idx, 'length': end_idx - start_idx + 1}) 

print(f"🎬 분할된 연속 동작 세그먼트 총 개수: {len(segments)}개") 


# ==============================================================================
# 🎥 6. 군집별 세부 폴더 분리형 비디오 클리핑 및 저장 (기존 유지)
# ==============================================================================
print("🚀 추출된 구간별 비디오 클리핑 인코딩 시작...") 
first_img = cv2.imread(str(valid_frame_paths[0])) 
height, width, layers = first_img.shape 
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

for idx, seg in enumerate(segments): 
    label = seg['label'] 
    start = seg['start'] 
    end = seg['end'] 
    
    if label == -1: 
        cluster_folder_name = "cluster_noise" 
    else: 
        cluster_folder_name = f"cluster_{label}" 
        
    target_cluster_dir = CLIPS_OUT_DIR / cluster_folder_name 
    target_cluster_dir.mkdir(parents=True, exist_ok=True) 
    
    filename = f"seg{idx+1:02d}_f{start}to{end}.mp4" 
    clip_path = target_cluster_dir / filename 
    
    video_writer = cv2.VideoWriter(str(clip_path), fourcc, 30.0, (width, height)) 
    
    for f_idx in range(start, end + 1): 
        img = cv2.imread(str(valid_frame_paths[f_idx])) 
        if img is not None: 
            video_writer.write(img) 
            
    video_writer.release() 

print(f"🎥 군집별 분리 저장 완료된 클립 폴더: {CLIPS_OUT_DIR}") 


# ==============================================================================
# 🎯 7. .ass 자막 기반 Ground Truth 매핑 (기존 유지)
# ==============================================================================
ASS_PATH = Path(paths.get('ass', FRAME_DIR.parent / "p01_gross_motor_4.ass"))
print(f"\n📂 정답 자막(Ground Truth) 파일 로드 중: {ASS_PATH}")

gt_labels = ["Unknown/Rest"] * len(valid_frame_paths)

if not ASS_PATH.exists():
    print(f"⚠️ 경고: {ASS_PATH} 파일이 존재하지 않아 자막 매핑을 건너뜁니다.")
else:
    with open(ASS_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        if line.startswith("Dialogue:"):
            parts = line.split(",", 9)
            if len(parts) < 10: 
                continue
                
            start_ts = parts[1] 
            end_ts = parts[2]   
            text_content = parts[9].strip() 
            
            if not text_content: 
                continue
                
            start_frame = timestamp_to_frame_idx(start_ts)
            end_frame = timestamp_to_frame_idx(end_ts)
            
            for idx, f_path in enumerate(valid_frame_paths):
                frame_num_match = re.search(r'(\d+)\.(jpg|png)$', f_path.name)
                if frame_num_match:
                    actual_frame_idx = int(frame_num_match.group(1))
                    
                    if start_frame <= actual_frame_idx <= end_frame:
                        gt_labels[idx] = text_content

    print(f"📊 Ground Truth 매핑 완료. 매칭된 행동 목록 요약: {dict(Counter(gt_labels))}")

# ==============================================================================
# 🖼️ 8. 3가지 차원 축소 결과별 Ground Truth 개별 시각화 및 저장 (수정판)
# ==============================================================================
unique_gt_list = sorted(list(set(gt_labels)))

# 💡 변수명 오류 수정: embedding_2d 대신 정확히 연산된 embedding_umap을 전달합니다.
visualization_tasks = {
    "UMAP": (embedding_umap, SAVE_DIR / "dbscan_cluster_umap.png"), 
    "t-SNE": (embedding_tsne, SAVE_DIR / "dbscan_cluster_tsne.png"),
    "PCA": (embedding_pca, SAVE_DIR / "dbscan_cluster_pca.png")
}

for name, (embedding, out_path) in visualization_tasks.items():
    print(f"🎨 {name} 기반 잠재 공간 그래프 렌더링 중...")
    
    plt.figure(figsize=(14, 10))
    cmap = plt.get_cmap('tab20', len(unique_gt_list))
    
    # 정답 라벨별 산점도 레이어 추가
    for idx, label_name in enumerate(unique_gt_list):
        mask = [l == label_name for l in gt_labels]
        plt.scatter(
            embedding[mask, 0], embedding[mask, 1],
            label=label_name, color=cmap(idx), s=25, alpha=0.8, edgecolors='none'
        )

    # 시간 흐름 선 연결
    plt.plot(embedding[:, 0], embedding[:, 1], color='black', linewidth=0.4, alpha=0.2, zorder=1)

    # 시작점/끝점 마킹
    plt.scatter(embedding[0, 0], embedding[0, 1], color='red', s=140, marker='*', label='START_POINT', zorder=5)
    plt.scatter(embedding[-1, 0], embedding[-1, 1], color='blue', s=140, marker='X', label='END_POINT', zorder=5)

    plt.title(f"🔑 Expert Ground Truth Mapping on Full-Frame Latent Space ({name})", fontsize=15, fontweight='bold')
    plt.xlabel(f"{name} Dimension 1", fontsize=11)
    plt.ylabel(f"{name} Dimension 2", fontsize=11)

    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0), shadow=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.35)

    # 고화질 이미지 저장 후 객체 닫기
    plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 [{name} 완료] 그래프가 디스크에 저장되었습니다: {out_path.name}")

print(f"\n✨ 모든 시각화 결과물 파일이 {SAVE_DIR} 폴더 내에 완벽하게 빌드되었습니다!")