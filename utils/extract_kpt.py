import json # JSON 처리를 위한 라이브러리입니다.
import numpy as np # 수치 연산 및 배열 생성을 위한 넘파이입니다.
from pathlib import Path # 경로 처리를 위한 라이브러리입니다.
from tqdm import tqdm # 진행 상태 확인을 위한 라이브러리입니다.
from typing import Union # 타입 힌트를 통해 코드의 안정성을 높이는 라이브러리입니다.


# ==========================================
# 함수 1: 원본 데이터 추출 (Extraction) - 구간(Segment) 지원 버전
# ==========================================
def extract_id_keypoints(
    json_dir: Union[str, Path],     # JSON 파일들이 있는 디렉토리 경로
    target_id: int,                 # 추출하고자 하는 특정 사람의 ID
    start_frame: int = 0,           # 추출 시작 프레임
    end_frame: Union[int, float, None] = None  # 추출 종료 프레임 (None 허용)
) -> np.ndarray:
    """
    JSON 디렉토리의 하위 폴더를 포함하여 특정 ID의 [x, y, score]를 추출합니다.
    """
    json_path = Path(json_dir)
    # 💡 [핵심] rglob으로 하위 폴더의 모든 json을 찾습니다.
    all_json_files = sorted(list(json_path.rglob("*.json"))) 
    
    # 💡 [에러 해결] end_frame이 None이면 무한대로 설정하여 int() 변환 에러를 방지합니다.
    actual_end = float('inf') if end_frame is None else float(end_frame)
    actual_start = float(start_frame)

    target_files = []
    for f in all_json_files:
        try:
            # 파일명(stem)이 숫자라고 가정하고 인덱스를 추출합니다.
            frame_idx = int(f.stem) 
            if actual_start <= frame_idx <= actual_end:
                target_files.append(f)
        except ValueError:
            continue 
    
    if not target_files:
        print(f"⚠️ [주의] 설정된 구간({start_frame}~{end_frame})에 해당하는 파일이 없습니다.")
        return np.zeros((0, 12, 3))

    raw_data = []
    for file in tqdm(target_files, desc=f"Extracting ID:{target_id}"):
        # 빈 파일 체크
        if file.stat().st_size == 0:
            raw_data.append(np.zeros((12, 3)))
            continue

        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raw_data.append(np.zeros((12, 3)))
            continue

        frame_data = np.zeros((12, 3))
        # instance_info 리스트에서 target_id 매칭
        for inst in data.get('instance_info', []):
            # instance_id 또는 id 키를 모두 확인합니다.
            inst_id = inst.get('instance_id') if inst.get('instance_id') is not None else inst.get('id')
            
            if inst_id == target_id:
                coords = np.array(inst.get('keypoints', []))
                scores = np.array(inst.get('keypoint_scores', [])).reshape(-1, 1)
                
                if len(coords) >= 17:
                    # [17, 3] 형태로 합치기
                    full_kpts = np.hstack([coords[:, :2], scores])
                    # 얼굴(0-4) 제외, 신체(5-16) 추출 -> [12, 3]
                    frame_data = full_kpts[5:17, :] 
                break
        
        raw_data.append(frame_data)

    return np.array(raw_data)

# ==========================================
# 함수 2: 데이터 정규화 (12 Keypoints 대응)
# ==========================================
def normalize_skeleton_array(data_array):
    """
    (N, 12, 3) 형태의 배열을 받아 정규화합니다.
    (인덱스 주의: 원래 5-16번이 0-11번으로 당겨짐)
    """
    norm_data = data_array.copy().astype(float)
    
    for i in range(len(norm_data)):
        kpts = norm_data[i] # 현재 프레임 (12, 3)
        
        if np.all(kpts == 0):
            continue
        
        # 1. 중앙점 계산: 골반 중심 (6번, 7번 중점)
        hip_center = (kpts[6, :2] + kpts[7, :2]) / 2.0
        
        # 2. 기준 거리 계산: 몸통 길이 (어깨 중점과 골반 중점 사이 거리)
        shoulder_center = (kpts[0, :2] + kpts[1, :2]) / 2.0
        torso_length = np.linalg.norm(shoulder_center - hip_center)
        
        if torso_length > 1e-6:
            norm_data[i, :, :2] = (kpts[:, :2] - hip_center) / torso_length
            
    return norm_data
import json
from pathlib import Path

# ==========================================
# 함수 3: 12kpt 17kpt와 동일한 형식으로 덮어쓰기 (다른 ID 유지)
# ==========================================
def save_12kpt_to_17kpt_json(
    src_dir, 
    output_dir, 
    kpt_array, 
    target_id,
    start_frame=0
):
    src_path = Path(src_dir)
    dst_path = Path(output_dir)
    
    # 💡 [수정] rglob을 사용하여 하위 폴더의 모든 json을 가져와 정렬합니다.
    json_files = sorted(list(src_path.rglob('*.json')), key=lambda x: int(''.join(filter(str.isdigit, x.stem))))
    
    if start_frame >= len(json_files):
        print(f"⚠️ 에러: 시작 프레임({start_frame})이 전체 파일 수({len(json_files)})를 벗어납니다.")
        return

    print(f"🚀 {json_files[start_frame].relative_to(src_path)} 파일부터 병합을 시작합니다...")

    processed_count = 0
    for i in range(len(kpt_array)):
        file_idx = start_frame + i
        if file_idx >= len(json_files):
            print(f"⚠️ 매칭할 원본 파일이 부족하여 조기 종료합니다. (인덱스: {file_idx})")
            break
            
        json_file = json_files[file_idx]
        # 💡 [수정] 원본의 상대 경로를 계산하여 저장 경로를 생성합니다 (폴더 구조 유지).
        rel_path = json_file.relative_to(src_path)
        save_path = dst_path / rel_path
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for inst in data.get('instance_info', []):
                inst_id = inst.get('instance_id') if inst.get('instance_id') is not None else inst.get('id') 
                
                if inst_id == target_id:
                    for k in range(12):
                        json_idx = k + 5
                        inst['keypoints'][json_idx][0] = float(kpt_array[i, k, 0]) 
                        inst['keypoints'][json_idx][1] = float(kpt_array[i, k, 1]) 
                    break 
            
            # 💡 [수정] 저장 폴더가 없으면 생성 후 저장
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            
            processed_count += 1
        except Exception as e:
            print(f"⚠️ {json_file.name} 처리 중 에러: {e}")

    print(f"✅ 완료: 총 {processed_count}개 파일 변환 완료!")

# ==========================================
# 함수 4: 12kpt 17kpt와 동일한 형식으로 덮어쓰기 (patient_id 만 유지)
# ==========================================
def save_patient_only_12_to_17(
    src_dir, 
    output_dir, 
    kpt_array, 
    patient_id, 
    start_frame=0
):
    src_path = Path(src_dir)
    dst_path = Path(output_dir)

    # 💡 [수정] rglob 사용 및 숫자 기준 정렬
    json_files = sorted(list(src_path.rglob('*.json')), key=lambda x: int(''.join(filter(str.isdigit, x.stem))))
    
    if start_frame >= len(json_files):
        print(f"⚠️ 에러: 시작 프레임({start_frame})이 전체 파일 수({len(json_files)})를 벗어납니다.")
        return

    print(f"🚀 {json_files[start_frame].relative_to(src_path)} 파일부터 병합을 시작합니다...")

    processed_count = 0
    for i in range(len(kpt_array)):
        file_idx = start_frame + i
        if file_idx >= len(json_files):
            print(f"⚠️ 매칭할 원본 파일이 부족하여 조기 종료합니다. (인덱스: {file_idx})")
            break
            
        json_file = json_files[file_idx]
        rel_path = json_file.relative_to(src_path)
        save_path = dst_path / rel_path
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            filtered_instances = []
            for inst in data.get('instance_info', []):
                inst_id = inst.get('instance_id') if inst.get('instance_id') is not None else inst.get('id')
                
                if inst_id == patient_id:
                    for k in range(12):
                        json_idx = k + 5
                        inst['keypoints'][json_idx][0] = float(kpt_array[i, k, 0])
                        inst['keypoints'][json_idx][1] = float(kpt_array[i, k, 1])
                    filtered_instances.append(inst)
                    break
            
            data['instance_info'] = filtered_instances
            
            # 💡 [수정] 저장 폴더 생성 및 저장
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            
            processed_count += 1
        except Exception as e:
            print(f"⚠️ {json_file.name} 처리 중 에러: {e}")

    if processed_count > 0:
        print(f"✅ 완료: 총 {processed_count}개 파일 필터링 및 변환 완료!")
    else:
        print("⚠️ 완료: 처리된 파일이 없습니다.")