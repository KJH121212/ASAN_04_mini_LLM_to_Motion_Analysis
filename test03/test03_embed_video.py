# ==============================================================================
# 🔍 고도화된 Keypoint 전처리 및 초경량 UMAP 시각화
# - 얼굴 관절(0~4) 제외: 어깨부터 발목까지 12개 관절(24차원)만 사용
# - Confidence Score 기반 필터링: 0.05 이하는 노이즈로 간주하고 중앙값(0.5)으로 처리
# - FAISS DB 저장 없이 곧바로 메모리에 올려 UMAP 궤적을 확인합니다.
# ==============================================================================

import json # JSON 파싱용
import numpy as np # 행렬 및 벡터 연산용
from pathlib import Path # 파일 경로 관리용
import umap # 고차원 데이터 시각화(차원 축소)용
import matplotlib.pyplot as plt # 궤적 그래프 출력용
from tqdm import tqdm # 콘솔 진행률 표시용

# ==============================================================================
# 🛠️ 핵심 전처리 함수: 얼굴 제외 & Confidence Score 필터링 적용
# ==============================================================================
def get_refined_keypoint_vector(json_path, target_patient_id):
    """
    JSON에서 관절 좌표를 읽어와 다음의 전처리를 수행합니다:
    1. 얼굴 관절(0번~4번) 제외 -> 5번~16번(총 12개)만 사용
    2. Confidence Score 0.05 이하인 관절은 (0.5, 0.5)로 중앙값 패딩 (노이즈 방지)
    3. BBox를 기준으로 좌표를 0~1 사이로 정규화 (크기/위치 불변성 확보)
    => 최종 24차원(12*2)의 1D 벡터 반환
    """
    if not json_path.exists():
        return None 
        
    with open(json_path, 'r') as f:
        keypoint_data = json.load(f)
        
    # 타겟 환자 인스턴스 검색
    target_instance = next((inst for inst in keypoint_data.get('instance_info', []) 
                            if inst.get('instance_id') == target_patient_id), None)
                            
    if target_instance is None:
        return None # 환자가 없으면 스킵
        
    bbox = target_instance.get('bbox')
    x_min, y_min, x_max, y_max = bbox
    
    # BBox 너비/높이 계산 (0이 되는 것 방지)
    width = max(x_max - x_min, 1e-5) 
    height = max(y_max - y_min, 1e-5)
    
    keypoints = target_instance.get('keypoints', []) # 좌표 [x, y] 리스트
    scores = target_instance.get('keypoint_scores', []) # 각 관절의 신뢰도 점수 리스트
    
    if len(keypoints) < 17 or len(scores) < 17:
        return None # 데이터가 손상된 프레임 스킵
    
    refined_vector = []
    
    # 💡 5번(왼쪽 어깨)부터 16번(오른쪽 발목)까지만 사용 (얼굴 0~4번 제외)
    for i in range(5, 17): 
        x, y = keypoints[i]
        score = scores[i]
        
        # 💡 Confidence Score 필터링 (0.05 이하)
        if score <= 0.05:
            # 신뢰도가 너무 낮으면 노이즈가 될 수 있으므로, 해당 관절이 가려졌다고 판단.
            # 튀는 값을 막기 위해 BBox의 상대적 중앙인 (0.5, 0.5)를 부여합니다.
            nx = 0.5
            ny = 0.5
        else:
            # 신뢰도가 높으면 정상적으로 BBox 기준 0~1 정규화를 수행합니다.
            nx = (x - x_min) / width
            ny = (y - y_min) / height
            
            # (선택) 정규화 후 혹시라도 BBox를 살짝 벗어난 값이 있다면 0~1 사이로 잘라줍니다(Clipping).
            nx = np.clip(nx, 0.0, 1.0)
            ny = np.clip(ny, 0.0, 1.0)
            
        refined_vector.extend([nx, ny]) # [nx5, ny5, nx6, ny6, ...] 순으로 이어붙임
        
    return np.array(refined_vector, dtype=np.float32) # 최종 24차원 배열 반환

# ==============================================================================
# 🚀 메인 실행부
# ==============================================================================
if __name__ == "__main__":
    # 1. 경로 및 대상 설정 (사용자 환경에 맞게 하드코딩)
    KEYPOINT_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data_d03/02_KEYPOINTS/1_bayley/p01/p01_gross_motor_4")
    TARGET_ID = 1
    
    print("🔍 JSON 파일 탐색 중...")
    json_paths = sorted(list(KEYPOINT_DIR.rglob("*.json"))) # 모든 JSON을 시간 순으로 정렬하여 로드
    print(f"📊 총 {len(json_paths)}개의 JSON 파일을 찾았습니다.")

    # 2. 데이터 추출 및 전처리 (메모리)
    features = []
    
    for j_path in tqdm(json_paths, desc="고도화 전처리 및 좌표 추출 중"):
        pose_vector = get_refined_keypoint_vector(j_path, TARGET_ID)
        
        if pose_vector is not None:
            features.append(pose_vector)
            
    features = np.array(features)
    print(f"✨ 추출 완료! 24차원 벡터가 총 {features.shape[0]}개 모였습니다. (얼굴 제외, Confidence 필터 적용)")

    # 3. UMAP 차원 축소 (24D -> 2D)
    print("🎨 UMAP 차원 축소를 시작합니다...")
    # 포즈 벡터는 차원이 작고 데이터가 밀집해 있으므로 파라미터를 살짝 조정했습니다.
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, metric='euclidean', random_state=42) 
    embedding_2d = reducer.fit_transform(features)
    
    # 4. 시각화 및 저장
    plt.figure(figsize=(11, 8))
    
    # 시간 순서대로 색상이 변하도록 스캐터 플롯 생성
    sc = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                     c=range(len(embedding_2d)), cmap='plasma', s=15, alpha=0.7)
                     
    # 프레임 궤적을 잇는 얇은 선
    plt.plot(embedding_2d[:, 0], embedding_2d[:, 1], color='gray', linewidth=0.3, alpha=0.3)
    
    # 시작점과 끝점 강조
    plt.scatter(embedding_2d[0, 0], embedding_2d[0, 1], color='green', s=120, marker='*', label='START (Frame 0)', zorder=5)
    plt.scatter(embedding_2d[-1, 0], embedding_2d[-1, 1], color='red', s=120, marker='X', label=f'END (Frame {len(embedding_2d)-1})', zorder=5)
    
    plt.colorbar(sc, label="Timeline (Valid Frame Index)")
    plt.title("Refined Pose Trajectory (No Face, Conf > 0.05, 24D -> 2D)", fontsize=14, fontweight='bold')
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.4)
    
    save_path = "./refined_pose_umap_result.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 시각화 이미지 저장 완료: {save_path}")