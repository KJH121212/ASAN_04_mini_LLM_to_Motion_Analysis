# ==============================================================================
# 🦴 [실험 B-3] ST-GCN (시공간 그래프 합성곱 신경망) 비디오 클립 임베딩
# - 사람의 관절 구조를 물리적 그래프(Graph, Nodes & Edges)로 정의
# - 공간(Spatial) 그래프 합성곱 + 시간(Temporal) 1D 합성곱 동시 수행
# - [Batch, Channels(2), Time(16), Vertices(12)] 차원의 텐서를 256D 벡터로 압축
# ==============================================================================

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# ==============================================================================
# 📂 1. 데이터 경로 설정
# ==============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
print(f"📁 프로젝트 루트 경로: {PROJECT_ROOT}")

from utils.path_list_d03 import path_list_d03

DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data_d03")  
CSV_PATH = DATA_DIR / "metadata_v1.0.csv"
df = pd.read_csv(str(CSV_PATH))

# 저장 경로
SAVE_DIR = PROJECT_ROOT / "test07"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
TARGET_PATIENT_ID = 1

# ==============================================================================
# 🧠 2. ST-GCN 모델 아키텍처 (그래프 정의 및 합성곱 레이어)
# ==============================================================================

# 💡 [핵심] 12개 관절의 물리적 연결성(Graph Edges)을 정의합니다.
# 인덱스: 0(L어깨), 1(R어깨), 2(L팔꿈), 3(R팔꿈), 4(L손목), 5(R손목)
#        6(L골반), 7(R골반), 8(L무릎), 9(R무릎), 10(L발목), 11(R발목)
edges = [
    (0, 1), (6, 7), (0, 6), (1, 7), # 몸통 연결 (어깨-골반)
    (0, 2), (2, 4),                 # 왼쪽 팔 (어깨-팔꿈치-손목)
    (1, 3), (3, 5),                 # 오른쪽 팔
    (6, 8), (8, 10),                # 왼쪽 다리 (골반-무릎-발목)
    (7, 9), (9, 11)                 # 오른쪽 다리
]

# 인접 행렬(Adjacency Matrix) 생성 및 정규화
A = np.zeros((12, 12))
for i, j in edges:
    A[i, j] = A[j, i] = 1
A = A + np.eye(12) # 자기 자신과의 연결(Self-loop) 추가
D = np.diag(np.sum(A, axis=1)**(-0.5))
A_normalized = np.dot(np.dot(D, A), D) # 스펙트럴 그래프 정규화

class SpatialGraphConv(nn.Module):
    """공간(Spatial) 축에서 관절 간의 상호작용을 계산하는 그래프 합성곱"""
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.A = nn.Parameter(torch.tensor(A, dtype=torch.float32), requires_grad=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        # x 형태: [Batch, Channels, Time, Vertices]
        x = self.conv(x)
        # 아인슈타인 표기법(einsum)을 사용하여 그래프 행렬(A) 곱 수행
        x = torch.einsum('n c t v, v w -> n c t w', x, self.A)
        return x

class STGCNBlock(nn.Module):
    """공간(관절 연결)과 시간(프레임 흐름)을 동시에 압축하는 블록"""
    def __init__(self, in_channels, out_channels, A, stride=1):
        super().__init__()
        self.sgcn = SpatialGraphConv(in_channels, out_channels, A)
        # 시간(Time) 차원만 줄이는(stride) 1D 역할의 2D Conv
        self.tcn = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.bn(self.sgcn(x)))
        x = self.relu(self.bn(self.tcn(x)))
        return x

class STGCN_Encoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.stgcn_layers = nn.Sequential(
            STGCNBlock(2, 64, A_normalized),          # [B, 64, 16, 12]
            STGCNBlock(64, 128, A_normalized, stride=2), # [B, 128, 8, 12] (시간 축 축소)
            STGCNBlock(128, embed_dim, A_normalized, stride=2) # [B, 256, 4, 12]
        )
        # 모든 시간과 관절을 1개의 점으로 평균(Global Pool)
        self.pool = nn.AdaptiveAvgPool2d(1) 

    def forward(self, x):
        # 입력 x 형태: [Batch, C(2), T(16), V(12)]
        x = self.stgcn_layers(x) # -> [Batch, 256, 1, 1]
        x = self.pool(x)
        return x.view(x.size(0), -1) # -> [Batch, 256]

# ==============================================================================
# 🛠️ 3. 전처리 유틸리티 (형태 변환: [2, 16, 12] 텐서 생성)
# ==============================================================================
def timestamp_to_frame_idx(ts_str, fps=30.0):
    ts_str = ts_str.strip()
    if '.' not in ts_str: ts_str += ".00"
    time_part, ms_part = ts_str.split('.')
    t = datetime.strptime(time_part, "%H:%M:%S") if time_part.count(':') == 2 else datetime.strptime(time_part, "%M:%S")
    ms = int(ms_part) * 10 if len(ms_part) == 2 else int(ms_part)
    return int((t.hour * 3600 + t.minute * 60 + t.second + ms / 1000.0) * fps)

def get_all_clips(ass_path, total_frames):
    labeled_clips = []
    with open(ass_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("Dialogue:"):
                parts = line.split(",", 9)
                if len(parts) < 10: continue
                label = parts[9].strip()
                if label:
                    labeled_clips.append({"start": timestamp_to_frame_idx(parts[1]), "end": timestamp_to_frame_idx(parts[2]), "label": label})
    labeled_clips.sort(key=lambda x: x['start'])
    
    all_clips, current_frame = [], 0
    for clip in labeled_clips:
        if clip['start'] > current_frame:
            all_clips.append({"start": current_frame, "end": clip['start'] - 1, "label": "Unknown/Rest"})
        all_clips.append(clip)
        current_frame = max(current_frame, clip['end'] + 1)
    if current_frame < total_frames:
        all_clips.append({"start": current_frame, "end": total_frames - 1, "label": "Unknown/Rest"})
    return all_clips

def load_and_sample_stgcn_tensor(frame_paths, keypoint_dir, start, end, num_frames=16, conf_threshold=0.05):
    """
    관절 좌표를 보간한 뒤, 💡양어깨 기준 영점 조준(Root-Centering)을 수행하고, 
    ST-GCN 전용 규격인 [C(2), T(16), V(12)] 텐서로 변환합니다.
    """
    indices = np.linspace(start, end, num_frames, dtype=int)
    raw_coords = np.zeros((num_frames, 24)) 
    
    for t, idx in enumerate(indices):
        safe_idx = min(idx, len(frame_paths) - 1)
        json_path = keypoint_dir / f"{frame_paths[safe_idx].stem}.json"
        
        if json_path.exists():
            with open(json_path, 'r') as f:
                json_data = json.load(f)
                instance = next((i for i in json_data.get('instance_info', []) if i.get('instance_id') == TARGET_PATIENT_ID), None)
                if instance and len(instance.get('keypoints', [])) >= 17:
                    for v, kp in enumerate(instance['keypoints'][5:17]): # 12개 관절 추출
                        if kp[2] <= conf_threshold:
                            raw_coords[t, v*2:v*2+2] = np.nan
                        else:
                            raw_coords[t, v*2:v*2+2] = kp[0:2]
                else:
                    raw_coords[t, :] = np.nan
        else:
            raw_coords[t, :] = np.nan

    # 1. Pandas 선형 보간 처리
    df = pd.DataFrame(raw_coords).interpolate(method='linear', limit_direction='both').bfill().ffill().fillna(0)
    interpolated_coords = df.to_numpy(dtype=np.float32) # [T(16), 24]
    
    # ==============================================================================
    # 💡 2. Root-Centering (영점 조준) 핵심 로직 추가
    # ==============================================================================
    for t in range(num_frames):
        # 데이터가 아예 없는 프레임은 제외
        if np.all(interpolated_coords[t] == 0):
            continue
            
        # 양어깨의 중심 좌표 (Center X, Center Y) 계산
        # 인덱스 0,1: 왼쪽 어깨(x,y) / 인덱스 2,3: 오른쪽 어깨(x,y)
        center_x = (interpolated_coords[t, 0] + interpolated_coords[t, 2]) / 2.0
        center_y = (interpolated_coords[t, 1] + interpolated_coords[t, 3]) / 2.0
        
        # 모든 관절(12개)의 X좌표에서 center_x를, Y좌표에서 center_y를 뺌
        for v in range(12):
            interpolated_coords[t, v*2] -= center_x     # X축 원점 이동
            interpolated_coords[t, v*2 + 1] -= center_y # Y축 원점 이동
    # ==============================================================================
    
    # 3. ST-GCN 입력을 위한 차원 재구성 (Reshape & Permute)
    # [T(16), 12, 2] 로 분리 (12개 관절, 각 2차원(x,y))
    stgcn_tensor = interpolated_coords.reshape(num_frames, 12, 2)
    # [C(2), T(16), V(12)] 로 축 변환
    stgcn_tensor = np.transpose(stgcn_tensor, (2, 0, 1))
    
    return torch.tensor(stgcn_tensor).unsqueeze(0) # 배치(1) 추가 -> [1, 2, 16, 12]

# ==============================================================================
# 🚀 4. 모델 가동 및 순회 추출
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n🚀 Graph 기반 ST-GCN 인코더 가동 준비 중...")
model = STGCN_Encoder().to(device)
model.eval()

print(f"\n📊 총 {len(df)}명의 환자 데이터 처리를 시작합니다. (ST-GCN 모드)")

all_embeddings, all_labels = [], []
integrated_save_path = SAVE_DIR / "bayley_features.npz"

if integrated_save_path.exists():
    print(f"\n⏩ 이미 임베딩 파일이 존재합니다: {integrated_save_path}")
else:
    for target in tqdm(range(len(df)), desc="전체 진행률"):
        common_path = df.iloc[target]['common_path']
        if 'gross_motor' not in str(common_path): continue
            
        paths = path_list_d03(common_path)
        if 'ass' not in paths or not Path(paths['ass']).exists(): continue
            
        FRAME_DIR, KEYPOINT_DIR = Path(paths['frame']), Path(paths['keypoint'])
        if not FRAME_DIR.exists() or not KEYPOINT_DIR.exists(): continue
            
        frame_paths = sorted(list(FRAME_DIR.rglob("*.jpg")))
        if len(frame_paths) == 0: continue

        for clip in get_all_clips(Path(paths['ass']), len(frame_paths)):
            stgcn_tensor = load_and_sample_stgcn_tensor(
                frame_paths, KEYPOINT_DIR, clip['start'], clip['end'], num_frames=16
            ).to(device)
            
            with torch.no_grad():
                embedding = model(stgcn_tensor) # [1, 256] 추출
                all_embeddings.append(embedding.cpu().numpy().squeeze())
                all_labels.append(clip['label'])

    if len(all_embeddings) > 0:
        np.savez(integrated_save_path, embeddings=np.array(all_embeddings), labels=np.array(all_labels))
        print(f"\n✨ 성공! {len(all_labels)}개의 임베딩 저장 완료: {integrated_save_path}")