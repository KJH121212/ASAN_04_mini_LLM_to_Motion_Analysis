# ==============================================================================
# 🚀 1단계: 비디오 프레임 특징 추출 및 Vector DB(FAISS) 구축
# - BBox 기반 배경 마스킹 전처리 적용 (타겟 환자 포즈에만 집중)
# - ViT(Vision Transformer) 백본 모델을 통해 768차원 밀집 특징 벡터 추출
# - 로컬(.npy) 및 FAISS(.faiss) 데이터베이스에 캐싱
# ==============================================================================

import os
import sys
import json # JSON 파일을 읽기 위해 내장 json 모듈을 불러옵니다.
import pandas as pd # 메타데이터 처리를 위해 pandas를 불러옵니다.
import numpy as np # 행렬 및 배열 연산을 위해 numpy를 불러옵니다.
import torch # 딥러닝 모델 연산을 위해 PyTorch를 불러옵니다.
import cv2 # 이미지 처리 및 시각화 디버깅을 위해 OpenCV를 불러옵니다.
import faiss # 빠른 벡터 검색 및 DB 구축을 위해 faiss를 불러옵니다.
import umap # 고차원 벡터를 2차원으로 축소하기 위해 umap을 불러옵니다.
import matplotlib.pyplot as plt # 차원 축소 결과 시각화를 위해 불러옵니다.
from tqdm import tqdm # 반복문 진행률을 보기 위해 불러옵니다.
from transformers import ViTImageProcessor, ViTModel # 허깅페이스의 ViT 모델과 전처리기를 불러옵니다.
from pathlib import Path # 파일 및 디렉토리 경로 관리를 위해 불러옵니다.
from PIL import Image # ViT 모델 입력을 위한 이미지 포맷 변환용으로 불러옵니다.

# ==============================================================================
# 🛠️ 함수 정의 부
# ==============================================================================

def get_masked_patient_image(img_path, json_path, target_patient_id, debug=False):
    """
    특정 patient_id(instance_id)에 해당하는 BBox 영역만 남기고 나머지는 검은색으로 처리합니다.
    debug=True 일 경우, 마스킹된 이미지 위에 BBox와 Keypoint를 오버레이하여 시각화합니다.
    """
    img_np = cv2.imread(str(img_path)) # 이미지를 BGR 포맷의 NumPy 배열로 읽어옵니다.
    if img_np is None: 
        return None # 이미지 파일이 깨졌거나 없는 경우 None을 반환합니다.
        
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) # ViT 모델에 맞게 RGB 포맷으로 변환합니다.
    
    if not json_path.exists(): 
        # print(f"⚠️ 경고: JSON 파일이 없습니다 -> {json_path}") # 필요시 주석 해제하여 확인
        return None # JSON 파일이 없으면 처리를 중단합니다.
        
    with open(json_path, 'r') as f: 
        keypoint_data = json.load(f) # JSON 데이터를 파싱합니다.
        
    target_instance = None # 타겟 인스턴스를 저장할 변수를 초기화합니다.
    for instance in keypoint_data.get('instance_info', []): 
        if instance.get('instance_id') == target_patient_id: # ID가 일치하는지 확인합니다.
            target_instance = instance # 일치하는 인스턴스를 찾으면 저장합니다.
            break 
            
    if target_instance is None: 
        # print(f"⚠️ 경고: 프레임 {img_path.name}에서 patient_id {target_patient_id}를 찾을 수 없습니다.")
        return None # 해당 프레임에 타겟 환자가 없으면 None을 반환합니다.
        
    bbox = target_instance.get('bbox') # BBox 좌표를 가져옵니다.
    x_min, y_min, x_max, y_max = map(int, bbox) # 소수점 방지를 위해 정수형으로 변환합니다.
    
    masked_img = np.zeros_like(img_np) # 원본과 같은 크기의 검은색 도화지를 만듭니다.
    masked_img[y_min:y_max, x_min:x_max] = img_np[y_min:y_max, x_min:x_max] # BBox 영역만 복사합니다.
    
    if debug: # 디버깅 모드일 때만 실행됩니다.
        print(f"\n🐛 [DEBUG] 프레임: {img_path.name} | 타겟 ID: {target_patient_id} | BBox: {bbox}")
        debug_img = masked_img.copy() # 시각화를 위한 복사본을 생성합니다.
        cv2.rectangle(debug_img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2) # 빨간색 테두리를 그립니다.
        
        for kp in target_instance.get('keypoints', []): 
            cv2.circle(debug_img, (int(kp[0]), int(kp[1])), 4, (0, 255, 0), -1) # 초록색 관절 포인트를 그립니다.
            
        plt.figure(figsize=(8, 6)) # 피규어 창 크기를 설정합니다.
        plt.imshow(debug_img) # 이미지를 렌더링합니다.
        plt.title(f"Debug Overlay: {img_path.name}") # 타이틀을 지정합니다.
        plt.axis('off') # 눈금을 숨깁니다.
        plt.show() # 화면에 띄웁니다.
        
    return Image.fromarray(masked_img) # 전처리 완료된 NumPy 배열을 PIL Image 객체로 반환합니다.


# ==============================================================================
# 📂 1. 경로 및 데이터 설정
# ==============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT)) # 외부 모듈 임포트를 위해 프로젝트 루트를 시스템 경로에 추가합니다.
print(f"📁 프로젝트 루트 경로: {PROJECT_ROOT}")

SAVE_DIR = PROJECT_ROOT / "test02"
SAVE_DIR.mkdir(parents=True, exist_ok=True) # 저장 폴더가 없으면 생성합니다.

FAISS_PATH = SAVE_DIR / "bayley_gross_motor.faiss" # Vector DB 저장 경로입니다.
FEATURES_PATH = SAVE_DIR / "bayley_features.npy" # 특징 행렬 저장 경로입니다.

from utils.path_list_d03 import path_list_d03 # 사용자 정의 모듈을 불러옵니다.

DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data_d03")
CSV_PATH = DATA_DIR / "metadata_v1.0.csv" # 메타데이터 경로입니다.

df = pd.read_csv(str(CSV_PATH)) # CSV 파일을 데이터프레임으로 읽어옵니다.

target = 0
common_path = df.iloc[target]['common_path']
paths = path_list_d03(common_path) # 지정된 환자의 공통 경로 정보를 가져옵니다.

FRAME_DIR = Path(paths['frame'])
KEYPOINT_DIR = Path(paths['keypoint'])
target_patient_id = 1 # 타겟 환자의 ID를 설정합니다. (필요 시 df에서 동적으로 할당 가능)

print(f"🖼️ 프레임 데이터 위치: {FRAME_DIR}")

frame_paths = sorted(list(FRAME_DIR.rglob("*.jpg"))) # 폴더 내의 모든 jpg 파일을 정렬하여 리스트로 만듭니다.
print(f"📊 총 탐색된 프레임 수: {len(frame_paths)}장")

if len(frame_paths) == 0:
    raise FileNotFoundError("지정된 경로에 .jpg 파일이 없습니다. 경로를 다시 확인해주세요.")


# ==============================================================================
# 🚀 2. 벡터 캐시 확인 및 특징 추출
# ==============================================================================

if FAISS_PATH.exists() and FEATURES_PATH.exists(): # 이전에 추출해둔 데이터가 있는지 확인합니다.
    print("\n♻️ 이미 추출된 로컬 Vector DB 및 피처 파일이 존재합니다! 캐시에서 직접 불러옵니다.")
    features = np.load(str(FEATURES_PATH)) # npy 파일에서 특징 행렬을 로드합니다.
    index = faiss.read_index(str(FAISS_PATH)) # Vector DB를 로드합니다.
    print(f"📦 Vector DB 복원 완료! (등록된 벡터 수: {index.ntotal}개, 피처 크기: {features.shape})")

else:
    print("\n⚠️ 캐시된 데이터가 없습니다. 새로운 특징 추출을 시작합니다.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU 사용 가능 여부를 확인합니다.
    print(f"💻 사용 중인 디바이스: {device}")

    # ViT 모델 로드
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224') # 프로세서를 불러옵니다.
    model = ViTModel.from_pretrained('google/vit-base-patch16-224').to(device) # 백본 모델을 로드합니다.
    model.eval() # 추론 모드로 변경합니다.

    features = [] # 추출된 벡터를 담을 리스트입니다.
    valid_frame_paths = [] # 정상적으로 처리된 프레임의 경로를 보관합니다.
    debug_done = False # 디버깅 이미지를 한 번만 띄우기 위한 플래그입니다.

    print(f"🚀 타겟 환자(ID: {target_patient_id}) BBox 마스킹 후 임베딩 추출 중...")
    
    with torch.no_grad(): # 메모리 절약을 위해 그래디언트 연산을 끕니다.
        for f_path in tqdm(frame_paths): # 모든 프레임을 순회합니다.
            
            relative_path = f_path.relative_to(FRAME_DIR) # 원본 프레임 폴더로부터의 하위 경로 구조를 유지합니다.
            json_relative_path = relative_path.with_suffix('.json') # 확장자만 .json으로 교체합니다.
            j_path = KEYPOINT_DIR / json_relative_path # 최종 JSON 절대 경로를 결합합니다.
            
            should_debug = not debug_done # 아직 디버깅을 안 했다면 True가 됩니다.
            
            # 마스킹 전처리 수행 (환자가 없으면 None 반환)
            masked_pil_image = get_masked_patient_image(f_path, j_path, target_patient_id, debug=should_debug)
            
            if masked_pil_image is None: 
                continue # 환자가 해당 프레임에 없으므로 모델 연산을 패스합니다.
                
            if should_debug: 
                debug_done = True # 디버깅 이미지가 성공적으로 출력되었으므로 플래그를 닫습니다.
                
            # ViT 모델 추론
            inputs = processor(images=masked_pil_image, return_tensors="pt").to(device) # 텐서로 변환하여 디바이스에 올립니다.
            outputs = model(**inputs) # 전방향 연산(Forward)을 수행합니다.
            cls_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy() # CLS 토큰만 가져와 NumPy로 변환합니다.
            
            features.append(cls_embedding) # 결과 리스트에 추가합니다.
            valid_frame_paths.append(f_path) # 성공한 경로를 추가합니다.

    # 추출 결과 저장
    features = np.array(features).astype('float32') # FAISS를 위해 float32 타입의 NumPy 배열로 만듭니다.
    print(f"✨ 임베딩 완료! 유효 프레임 수: {len(valid_frame_paths)}, 행렬 크기: {features.shape}")

    # FAISS (Vector DB) 구축
    dimension = features.shape[1] # 벡터의 차원(768)을 가져옵니다.
    faiss.normalize_L2(features) # 코사인 유사도 검색을 위해 L2 정규화를 수행합니다.
    index = faiss.IndexFlatIP(dimension) # 내적(Inner Product) 기반의 인덱스를 생성합니다.
    index.add(features) # 추출된 특징을 DB에 추가합니다.
    
    np.save(str(FEATURES_PATH), features) # 특징 행렬을 디스크에 저장합니다.
    faiss.write_index(index, str(FAISS_PATH)) # Vector DB를 디스크에 저장합니다.
    print(f"💾 피처 데이터 및 Vector DB 가 로컬에 저장되었습니다:\n  - Vector DB: {FAISS_PATH}\n  - Features: {FEATURES_PATH}")


# ==============================================================================
# 📊 3. 후처리 및 시각화 (시간 축 변위 분석 & UMAP)
# ==============================================================================

# 3-1. 시간 축 변위 분석 (Divider)
distances = []
for i in range(len(features) - 1): # 모든 인접 프레임 간의 유사도를 측정합니다.
    sim = np.dot(features[i], features[i+1]) # 정규화되었으므로 내적이 곧 코사인 유사도입니다.
    distances.append(1 - sim) # 거리는 1 - 유사도로 계산합니다.

threshold = np.percentile(distances, 99) # 전체 거리 분포 중 상위 1% 값을 임계치로 잡습니다.
divider_frames = [i for i, d in enumerate(distances) if d > threshold] # 움직임이 급격한 구간의 인덱스를 찾습니다.
print(f"🚨 동작이 급변하는 Divider 후보 프레임 (상위 1%): {divider_frames[:10]} ...")

# 3-2. UMAP 차원 축소
print("🎨 Latent Space 시각화를 위한 차원 축소 중 (UMAP)...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42) # UMAP 모델을 초기화합니다.
embedding_2d = reducer.fit_transform(features) # 768차원을 2차원으로 압축합니다.

# 3-3. Matplotlib 시각화 출력
plt.figure(figsize=(12, 8)) # 도화지 크기를 설정합니다.
sc = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                 c=range(len(embedding_2d)), cmap='viridis', s=15, alpha=0.6, label='Frames') # 프레임 순서대로 색상을 입혀 산점도를 그립니다.

plt.plot(embedding_2d[:, 0], embedding_2d[:, 1], color='gray', linewidth=0.5, alpha=0.4) # 점들을 선으로 이어 궤적을 표시합니다.

plt.scatter(embedding_2d[0, 0], embedding_2d[0, 1], color='red', s=100, marker='*', label='START', zorder=5) # 시작점을 붉은색 별로 표시합니다.
plt.scatter(embedding_2d[-1, 0], embedding_2d[-1, 1], color='blue', s=100, marker='X', label='END', zorder=5) # 종료점을 파란색 X로 표시합니다.

plt.colorbar(sc, label="Frame Timeline (Index)") # 우측에 타임라인 컬러바를 배치합니다.
plt.title("Frame-level Latent Space Trajectory (ViT + UMAP)", fontsize=14) # 제목을 작성합니다.
plt.xlabel("UMAP Dimension 1") # X축 라벨
plt.ylabel("UMAP Dimension 2") # Y축 라벨
plt.legend() # 범례 표시
plt.grid(True, linestyle='--', alpha=0.5) # 배경에 그리드를 깔아 가독성을 높입니다.

output_plot = "./frame_trajectory_result.png"
plt.savefig(str(output_plot), dpi=300, bbox_inches='tight') # 고해상도로 이미지를 저장합니다.
print(f"💾 시각화 결과 이미지 저장 완료: {output_plot}")
# plt.show() # 필요시 주석을 해제하여 리눅스 환경 밖(로컬 GUI)에서 직접 확인 가능합니다.