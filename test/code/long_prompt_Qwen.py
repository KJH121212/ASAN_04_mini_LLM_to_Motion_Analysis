import os
import sys
import json
import re
from pathlib import Path
import pandas as pd
import torch
import decord
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ==========================================
# 1. 환경 변수 및 경로 최적화 설정
# ==========================================
# 파이토치 메모리 단편화(OOM) 방지 옵션 켜기
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 프로젝트 최상위 폴더 경로 설정 및 sys.path 등록
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

print(f"📁 프로젝트 루트 경로: {PROJECT_ROOT}")

# 데이터 디렉토리 및 메타데이터 CSV, 매뉴얼(.md) 경로 정의
DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data_d03")
CSV_PATH = DATA_DIR / "metadata_v1.1.csv"
MANUAL_PATH = PROJECT_ROOT / "config" / "bayley_4th_motion.md"  # 지정하신 위치


# ==========================================
# 2. 파일 및 데이터 로드 함수 정의
# ==========================================
def load_bayley_manual(manual_path):
    """지정된 경로의 베일리 매뉴얼 .md 파일을 인코딩 예외 없이 안전하게 로드합니다."""
    if not manual_path.exists():
        raise FileNotFoundError(f"❌ 베일리 매뉴얼 파일을 찾을 수 없습니다: {manual_path}")
    
    with open(manual_path, "r", encoding="utf-8") as f:
        print(f"📖 베일리 4판 매뉴얼 로드 완료: {manual_path.name}")
        return f.read()


def load_qwen_model(model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
    """Qwen2.5-VL 모델 및 프로세서를 bfloat16 연산 모드로 안전하게 로드합니다."""
    print("🔄 Qwen2.5-VL-7B 모델 및 프로세서 초기화 중...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    print("✅ 모델 로드 완료!")
    return model, processor


# ==========================================
# 3. 핵심 영상 전처리 및 RAG 통합 채점 함수
# ==========================================
def analyze_and_score_bayley(video_path, retrieved_manual_md, model, processor, target_nframes=32):
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    native_fps = vr.get_avg_fps()
    video_duration = total_frames / native_fps
    
    print(f"🎞️ 원본 영상 정보: 총 {total_frames} 프레임 | 재생 시간 {video_duration:.2f}초")

    # 💡 치팅 없이 매뉴얼의 앵커(Primary -> Related) 구조를 강제하는 프롬프트
    prompt_text = f"""
    You are an expert clinical specialist scoring the Bayley Scales of Infant Development (4th Edition).
    Analyze the provided video segment and evaluate it strictly against the Official Test Manual Guideline provided below.

    ### [Official Test Manual Guideline (.md Context)]
    {retrieved_manual_md}

    ### [CRITICAL STEP-BY-STEP MISSION]
    1. **Identify the Primary Item:** Observe the child's movement in the video. Scan the provided manual (.md) and find the EXACT primary item section that matches this main action (e.g., Item 46).

    2. **Locate and Follow the 'Related Item' Link:**
    - Look at the `* **Related Item:**` field inside your identified Primary Item section. 
    - If it lists a number (e.g., "36"), you MUST search through the ENTIRE provided manual text above or below to find the block starting with `## Item 36`.
    - Read the specific "Scoring Criteria" for that related item and evaluate the video against it immediately. Do NOT leave "related_items_evaluation" empty if the related item exists in the manual text.

    Please output your final decision strictly in the following JSON format:
    ```json
    {{
    "primary_item": {{
        "item_id": "The exact main item identifier matched from the video (e.g., Item 46)",
        "item_name": "The title of the primary item",
        "scoring_decision": "Score based on the manual (e.g., 2, 1, or 0)",
        "clinical_justification": "Detailed medical explanation mapping video parameters directly to this primary item's criteria."
    }},
    "related_items_evaluation": [
        {{
        "item_id": "The related item identifier found strictly within the Primary Item's related field (e.g., Item 36)",
        "item_name": "The title of the related item found from the manual",
        "scoring_decision": "Score based on the related item's criteria (e.g., 2, 1, or 0)",
        "clinical_justification": "Detailed medical explanation mapping video parameters directly to this related item's specific criteria."
        }}
    ],
    "overall_behavioral_synthesis": "A cohesive narrative describing the full continuous movement flow of the child from start to finish within the {video_duration:.2f} seconds."
    }}
    """
    # Qwen 멀티모달 포맷 메시지 구조화
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "fps": target_nframes / video_duration},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # (이후 전처리 및 모델 생성 부는 기존과 동일)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1536)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

    return output_text[0]

# ==========================================
# 4. 메인 실행 제어부
# ==========================================

from utils.path_list_d03 import path_list_d03
    # 1. 메타데이터 파일 확인 및 데이터 로드
if not CSV_PATH.exists():
    raise FileNotFoundError(f"❌ 메타데이터 CSV를 찾을 수 없습니다: {CSV_PATH}")
    
df = pd.read_csv(str(CSV_PATH))

# 예시 타겟 (0번 인덱스 환자 샘플) 설정
target_idx = 0
common_path = df.iloc[target_idx]['common_path']
paths = path_list_d03(common_path)

# 폴더 내부 .mp4 파일 리스트화 및 첫 번째 비디오 파일 잡기
video_list = sorted(list(paths['split_video'].glob("*.mp4")))
if not video_list:
    raise FileNotFoundError(f"❌ '{paths['split_video']}' 폴더 내에 mp4 파일이 존재하지 않습니다.")

video_path = str(video_list[0])
print(f"🎯 실시간 분석 대상 비디오 지정: {Path(video_path).name}")

# 2. 로컬 config 폴더에서 베일리 매뉴얼 텍스트 전체 로드
bayley_manual_content = load_bayley_manual(MANUAL_PATH)

# 3. 인공지능 모델 및 프로세서 빌드
model, processor = load_qwen_model()

# 4. 고정 32프레임 기반 매뉴얼 융합 통합 채점 파이프라인 구동
final_report = analyze_and_score_bayley(
    video_path=video_path, 
    retrieved_manual_md=bayley_manual_content, 
    model=model, 
    processor=processor, 
    target_nframes=32
)

# 5. 최종 리포트 원문 출력
print("\n==================== [🎖️ Qwen2.5-VL RAG 통합 채점 결과 보고서] ====================\n")
print(final_report)
print("\n====================================================================================\n")