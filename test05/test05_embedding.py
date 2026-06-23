# ==============================================================================
# 🎬 [실험 B-2] BBox 마스킹 전처리 기반 Video Clip 임베딩 (MViT v2)
# - 타겟 환자의 BBox 외 영역을 검은색으로 마스킹 (배경 노이즈 제거)
# - df 내 gross_motor 영상 순회 및 .ass 파일 기반 클립 분할 (Rest 포함)
# - 16프레임(224x224) 샘플링 후 MViT_v2_s 인코더를 통해 768D 벡터 통합 저장
# ==============================================================================

import os
import sys
import json
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

# B-2 실험 결과를 저장할 전용 폴더 설정
SAVE_DIR = PROJECT_ROOT / "test05"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 🛠️ 2. 유틸리티 함수 (마스킹 로직 추가)
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

def get_masked_image(img_path, json_path, target_id):
    """BBox 영역만 남기고 배경을 검은색으로 마스킹하는 핵심 함수"""
    img_np = cv2.imread(str(img_path))
    if img_np is None: return None
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    
    if not json_path.exists(): return None
        
    with open(json_path, 'r') as f:
        keypoint_data = json.load(f)
        
    target_instance = next((inst for inst in keypoint_data.get('instance_info', []) 
                            if inst.get('instance_id') == target_id), None)
                            
    if target_instance is None: return None
        
    bbox = target_instance.get('bbox')
    x_min, y_min, x_max, y_max = map(int, bbox)
    
    # 예외 처리: BBox가 화면 밖을 벗어나는 경우 대비
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(img_np.shape[1], x_max), min(img_np.shape[0], y_max)
    
    # 원본과 같은 크기의 검은 도화지에 BBox 영역만 복사
    masked_img = np.zeros_like(img_np)
    masked_img[y_min:y_max, x_min:x_max] = img_np[y_min:y_max, x_min:x_max]
    
    return masked_img


def load_and_sample_masked_clip(frame_paths, keypoint_dir, start_idx, end_idx, target_patient_id, num_frames=16, target_size=(224, 224)):
    """클립 내 프레임을 균등 샘플링하고 마스킹 전처리를 거쳐 텐서로 반환"""
    indices = np.linspace(start_idx, end_idx, num_frames, dtype=int)
        
    frames = []
    for idx in indices:
        safe_idx = min(idx, len(frame_paths) - 1)
        f_path = frame_paths[safe_idx]
        
        # 대응하는 JSON 파일 경로 매핑
        json_path = keypoint_dir / f"{f_path.stem}.json"
        
        # 💡 [수정 완료] 전역 변수 대신 입력받은 target_patient_id를 정확히 사용
        masked_img = get_masked_image(f_path, json_path, target_patient_id)
        
        if masked_img is None:
            # 타겟 환자가 없거나 JSON이 깨진 경우, 완전히 검은 화면 추가
            masked_img = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        else:
            masked_img = cv2.resize(masked_img, target_size) 
        
        # 정규화 (Kinetics400 통계값)
        img = masked_img.astype(np.float32) / 255.0
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
model = video_models.mvit_v2_s(weights=video_models.MViT_V2_S_Weights.KINETICS400_V1)
model.head = torch.nn.Identity() # 768D 벡터 추출을 위해 분류기 헤드 제거
model.eval()
model.to(device)


# ==============================================================================
# 🔄 4. DataFrame 전체 순회 및 통합 임베딩 처리 (.ass 및 gross_motor 필터링)
# ==============================================================================
print(f"\n📊 총 {len(df)}명의 환자(Target) 데이터 처리를 시작합니다. (배경 마스킹 활성화)")

all_embeddings = []
all_labels = []

integrated_save_path = SAVE_DIR / "bayley_features.npz"

if integrated_save_path.exists():
    print(f"\n⏩ 이미 통합 임베딩 파일이 존재합니다. 연산을 건너뜁니다: {integrated_save_path}")
else:
    for target in tqdm(range(len(df)), desc="전체 진행률"):
        common_path = df.iloc[target]['common_path']
        
        # 💡 [핵심 추가] 현재 타겟 환자의 ID를 루프 안에서 동적으로 가져옵니다.
        current_patient_id = df.iloc[target]['patient_id']
        
        if 'gross_motor' not in str(common_path):
            continue
            
        paths = path_list_d03(common_path)
        
        # 필터링: 자막(.ass) 파일 존재 여부
        if 'ass' not in paths or not Path(paths['ass']).exists():
            continue
            
        ASS_PATH = Path(paths['ass'])
        FRAME_DIR = Path(paths['frame'])
        KEYPOINT_DIR = Path(paths['keypoint']) # 💡 마스킹을 위해 Keypoint 디렉토리 추가
        
        # 폴더 검증 (프레임 및 키포인트 JSON 폴더 필수)
        if not FRAME_DIR.exists() or not KEYPOINT_DIR.exists():
            continue
            
        frame_paths = sorted(list(FRAME_DIR.rglob("*.jpg")))
        total_frames = len(frame_paths)
        if total_frames == 0: continue

        # 클립 파싱 (Rest 구간 포함)
        clips_info = get_all_clips_including_rest(ASS_PATH, total_frames)
        
        for clip in clips_info:
            # 💡 [여기 수정] 파라미터로 current_patient_id를 넘겨줍니다.
            video_tensor = load_and_sample_masked_clip(
                frame_paths, KEYPOINT_DIR, clip['start'], clip['end'], current_patient_id, target_size=(224, 224)
            )
            
            if video_tensor is not None:
                video_tensor = video_tensor.to(device)
                
                with torch.no_grad():
                    embedding = model(video_tensor) # 형태: [1, 768]
                    all_embeddings.append(embedding.cpu().numpy().squeeze())
                    all_labels.append(clip['label'])

    # ==============================================================================
    # 💾 5. 루프 종료 후 일괄 저장
    # ==============================================================================
    if len(all_embeddings) > 0:
        np.savez(
            integrated_save_path, 
            embeddings=np.array(all_embeddings), 
            labels=np.array(all_labels)
        )
        print(f"\n✨ 성공! 총 {len(all_labels)}개의 'Masked' 클립 임베딩이 일괄 저장되었습니다.")
        print(f"📁 저장 위치: {integrated_save_path}")
    else:
        print("\n⚠️ 조건에 맞는 클립이 하나도 추출되지 않았습니다.")