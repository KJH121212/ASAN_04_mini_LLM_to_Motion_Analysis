import json
import re
import numpy as np
import cv2
import umap
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
from collections import Counter

# ==============================================================================
# 1. 핵심 전처리 함수: 얼굴 제외, Confidence 필터링, 정규화
# ==============================================================================
def get_refined_keypoint_vector(json_path, target_patient_id):
    if not json_path.exists(): return None
    with open(json_path, 'r') as f:
        keypoint_data = json.load(f)
    
    target_instance = next((inst for inst in keypoint_data.get('instance_info', []) 
                            if inst.get('instance_id') == target_patient_id), None)
    if not target_instance: return None
        
    bbox = target_instance.get('bbox')
    width, height = max(bbox[2]-bbox[0], 1e-5), max(bbox[3]-bbox[1], 1e-5)
    
    keypoints = target_instance.get('keypoints', [])
    scores = target_instance.get('keypoint_scores', [])
    
    refined_vector = []
    # 5번~16번 관절 사용 (총 12개, 24차원)
    for i in range(5, 17):
        x, y = keypoints[i]
        # 신뢰도 낮으면 (0.5, 0.5) 패딩
        if scores[i] <= 0.05:
            nx, ny = 0.5, 0.5
        else:
            nx = np.clip((x - bbox[0]) / width, 0.0, 1.0)
            ny = np.clip((y - bbox[1]) / height, 0.0, 1.0)
        refined_vector.extend([nx, ny])
    return np.array(refined_vector, dtype=np.float32)

def timestamp_to_frame_idx(ts_str, fps=30.0):
    t = datetime.strptime(ts_str.strip().split('.')[0], "%H:%M:%S")
    ms = int(ts_str.strip().split('.')[1]) * 10 if len(ts_str.strip().split('.')[1]) == 2 else int(ts_str.strip().split('.')[1])
    return int((t.hour*3600 + t.minute*60 + t.second + ms/100) * fps)

# ==============================================================================
# 2. 경로 설정 및 데이터 로드
# ==============================================================================
KEYPOINT_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data_d03/02_KEYPOINTS/1_bayley/p01/p01_gross_motor_4")
FRAME_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data_d03/01_FRAMES/1_bayley/p01/p01_gross_motor_4")
ASS_PATH = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/d03/ass/1_bayley/p01/p01_gross_motor_4.ass")
SAVE_DIR = Path("./skeleton_analysis_results")
SAVE_DIR.mkdir(exist_ok=True)

json_paths = sorted(list(KEYPOINT_DIR.rglob("*.json")))
features, valid_frame_paths = [], []

print("🚀 데이터 전처리 중...")
for j_path in tqdm(json_paths):
    vec = get_refined_keypoint_vector(j_path, 1)
    if vec is not None:
        features.append(vec)
        valid_frame_paths.append(FRAME_DIR / (j_path.stem + ".jpg"))

features = np.array(features)

# ==============================================================================
# 3. 군집화 및 차원 축소
# ==============================================================================
print("🤖 DBSCAN 및 차원 축소(UMAP, t-SNE, PCA) 수행 중...")
cluster_labels = DBSCAN(eps=0.2, min_samples=30).fit_predict(features)
embeddings = {
    "UMAP": umap.UMAP(n_neighbors=20, random_state=42).fit_transform(features),
    "t-SNE": TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(features),
    "PCA": PCA(n_components=2).fit_transform(features)
}

# ==============================================================================
# 4. 정답 라벨(ASS) 매핑 및 시각화
# ==============================================================================
gt_labels = ["Unknown/Rest"] * len(valid_frame_paths)
if ASS_PATH.exists():
    with open(ASS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("Dialogue:"):
                parts = line.split(",", 9)
                start_f, end_f = timestamp_to_frame_idx(parts[1]), timestamp_to_frame_idx(parts[2])
                for idx, f_path in enumerate(valid_frame_paths):
                    f_num = int(re.search(r'(\d+)\.(jpg|png)$', f_path.name).group(1))
                    if start_f <= f_num <= end_f: gt_labels[idx] = parts[9].strip()

unique_gt = sorted(list(set(gt_labels)))
for name, emb in embeddings.items():
    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap('tab20', len(unique_gt))
    for i, lbl in enumerate(unique_gt):
        mask = [l == lbl for l in gt_labels]
        plt.scatter(emb[mask, 0], emb[mask, 1], label=lbl, color=cmap(i), s=15, alpha=0.6)
    plt.title(f"Ground Truth Mapping ({name})"); plt.legend(); plt.savefig(SAVE_DIR / f"gt_{name}.png")
    plt.close()

# ==============================================================================
# 5. 비디오 클리핑 (군집별 폴더 자동 분할)
# ==============================================================================
print("🎥 비디오 클리핑 시작...")
segments = []
start, cur = 0, cluster_labels[0]
for i in range(1, len(cluster_labels)):
    if cluster_labels[i] != cur:
        if (i - start) >= 30: segments.append({'label': cur, 'start': start, 'end': i-1})
        start, cur = i, cluster_labels[i]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
for seg in segments:
    folder = SAVE_DIR / "clips" / f"cluster_{seg['label']}"
    folder.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(folder / f"seg_{seg['start']}_{seg['end']}.mp4"), fourcc, 30.0, (1920, 1080))
    for i in range(seg['start'], seg['end'] + 1):
        img = cv2.imread(str(valid_frame_paths[i]))
        if img is not None: out.write(img)
    out.release()

print("✨ 모든 작업 완료!")