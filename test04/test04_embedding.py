# ==============================================================================
# 🎬 [실험 B-1] PyTorch 비디오 인코더 클립 임베딩 자동화 (ASS 파일 필수 조건 추가)
# - df 내 모든 영상을 순회하되, .ass 파일이 존재하는 Target만 처리
# - 정답 자막이 없는 빈 시간대는 'Unknown/Rest' 클립으로 자동 추가
# - 16프레임 추출 후 3D CNN 인코더로 비디오 벡터 압축 및 저장
# ==============================================================================

import os
import sys
import pandas as pd
import numpy as np
import cv2
import torch
import torchvision.models.video as video_models
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# ==============================================================================
# 📂 1. 초기 경로 및 Data 설정
# ==============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
print(f"📁 프로젝트 루트 경로: {PROJECT_ROOT}")

from utils.path_list_d03 import path_list_d03

DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data_d03")  
CSV_PATH = DATA_DIR / "metadata_v1.0.csv"
df = pd.read_csv(str(CSV_PATH))

# 임베딩 결과를 저장할 전용 폴더 설정
SAVE_DIR = PROJECT_ROOT / "test04"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 🛠️ 2. 유틸리티 함수
# ==============================================================================
def timestamp_to_frame_idx(ts_str, fps=30.0):
    ts_str = ts_str.strip()
    if '.' not in ts_str: ts_str += ".00"
    time_part, ms_part = ts_str.split('.')
    t = datetime.strptime(time_part, "%H:%M:%S") if time_part.count(':') == 2 else datetime.strptime(time_part, "%M:%S")
    ms = int(ms_part) * 10 if len(ms_part) == 2 else int(ms_part)
    return int((t.hour * 3600 + t.minute * 60 + t.second + ms / 1000.0) * fps)

def get_all_clips_including_rest(ass_path, total_frames):
    labeled_clips = []
    with open(ass_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("Dialogue:"):
                parts = line.split(",", 9)
                if len(parts) < 10: continue
                start_f = timestamp_to_frame_idx(parts[1])
                end_f = timestamp_to_frame_idx(parts[2])
                label = parts[9].strip()
                if label:
                    labeled_clips.append({"start": start_f, "end": end_f, "label": label})
    
    labeled_clips.sort(key=lambda x: x['start'])
    
    all_clips = []
    current_frame = 0
    
    for clip in labeled_clips:
        if clip['start'] > current_frame:
            all_clips.append({"start": current_frame, "end": clip['start'] - 1, "label": "Unknown/Rest"})
        all_clips.append(clip)
        current_frame = max(current_frame, clip['end'] + 1)
        
    if current_frame < total_frames:
        all_clips.append({"start": current_frame, "end": total_frames - 1, "label": "Unknown/Rest"})
        
    return all_clips

def load_and_sample_video_clip(frame_paths, start_idx, end_idx, num_frames=16, target_size=(224, 224)):
    clip_length = end_idx - start_idx + 1
    indices = np.linspace(start_idx, end_idx, num_frames, dtype=int)
        
    frames = []
    for idx in indices:
        safe_idx = min(idx, len(frame_paths) - 1)
        img = cv2.imread(str(frame_paths[safe_idx]))
        if img is None: continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size) 
        
        # 정규화
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
        std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)
        img = (img - mean) / std
        
        frames.append(img)
        
    if len(frames) == 0: return None
        
    video_tensor = np.stack(frames, axis=0)
    video_tensor = np.transpose(video_tensor, (3, 0, 1, 2))
    return torch.tensor(video_tensor, dtype=torch.float32).unsqueeze(0)


# ==============================================================================
# 🧠 3. 최고 성능 비디오 트랜스포머 인코더 로드 (MViT v2)
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n🚀 고성능 PyTorch 비디오 인코더(MViT_v2_s) 모델 가동 준비 중...")
# Kinetics-400으로 학습된 MViT v2 모델 가중치 로드
model = video_models.mvit_v2_s(weights=video_models.MViT_V2_S_Weights.KINETICS400_V1)

# 💡 핵심: 분류기(Classifier) 역할을 하는 head를 잘라내어,
# 순수하게 768차원의 고품질 비디오 임베딩 벡터만 출력하도록 변경합니다.
model.head = torch.nn.Identity()

model.eval()
model.to(device)

# ==============================================================================
# 🔄 4. DataFrame 전체 순회 및 통합 임베딩 처리 (.ass 및 gross_motor 필터링)
# ==============================================================================
print(f"\n📊 총 {len(df)}명의 환자(Target) 데이터 처리를 시작합니다.")

# 💡 [핵심 추가] 전체 데이터를 하나로 모을 글로벌(통합) 리스트 생성
all_embeddings = []
all_labels = []

# 통합 저장될 최종 파일 경로 지정
integrated_save_path = SAVE_DIR / "integrated_B1_r3d18_embeddings.npz"

if integrated_save_path.exists():
    print(f"\n⏩ 이미 통합 임베딩 파일이 존재합니다. 연산을 건너뜁니다: {integrated_save_path}")
else:
    for target in tqdm(range(len(df)), desc="전체 진행률"):
        common_path = df.iloc[target]['common_path']
        
        # 1. 'gross_motor' 필터링
        if 'gross_motor' not in str(common_path):
            continue
            
        paths = path_list_d03(common_path)
        
        # 2. .ass 파일 필터링
        if 'ass' not in paths or not Path(paths['ass']).exists():
            continue
            
        ASS_PATH = Path(paths['ass'])
        FRAME_DIR = Path(paths['frame'])
        
        # 3. 프레임 폴더 검증
        if not FRAME_DIR.exists():
            continue
            
        frame_paths = sorted(list(FRAME_DIR.rglob("*.jpg")))
        total_frames = len(frame_paths)
        if total_frames == 0: continue

        # 4. 클립 파싱 (Rest 구간 포함)
        clips_info = get_all_clips_including_rest(ASS_PATH, total_frames)
        
        # 5. 임베딩 추출 및 '통합 리스트'에 병합
        for clip in clips_info:
            video_tensor = load_and_sample_video_clip(frame_paths, clip['start'], clip['end'])
            
            if video_tensor is not None:
                video_tensor = video_tensor.to(device)
                
                with torch.no_grad():
                    embedding = model(video_tensor) # 형태: [1, 512]
                    
                    # 💡 Target별 리스트가 아닌 전체 통합 리스트에 곧바로 Append
                    all_embeddings.append(embedding.cpu().numpy().squeeze())
                    all_labels.append(clip['label'])

    # ==============================================================================
    # 💾 5. 루프 종료 후, 거대한 하나의 파일로 일괄 저장
    # ==============================================================================
    if len(all_embeddings) > 0:
        np.savez(
            integrated_save_path, 
            embeddings=np.array(all_embeddings), 
            labels=np.array(all_labels)
        )
        print(f"\n✨ 성공! 총 {len(all_labels)}개의 클립 임베딩이 하나의 파일에 일괄 저장되었습니다.")
        print(f"📁 저장 위치: {integrated_save_path}")
    else:
        print("\n⚠️ 조건에 맞는 클립이 하나도 추출되지 않았습니다.")