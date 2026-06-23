# 텍스트 매칭 기반의 로컬 RAG 시스템
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
# 1. 환경 변수 및 경로 설정
# ==========================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data_d03")
CSV_PATH = DATA_DIR / "metadata_v1.1.csv"
MANUAL_PATH = PROJECT_ROOT / "config" / "bayley_4th_motion.md"


# ==========================================
# 2. 🔍 로컬 RAG(Retrieval) 시스템 함수 정의
# ==========================================
def build_manual_knowledge_base(manual_path):
    """
    매뉴얼(.md)을 읽어 각 Item ID(예: '36', '45', '46')를 키(Key)로 하는 
    딕셔너리 형태의 지식 저장소(Knowledge Base)를 빌드합니다.
    """
    if not manual_path.exists():
        raise FileNotFoundError(f"❌ 매뉴얼 파일이 없습니다: {manual_path}")
        
    with open(manual_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # '## Item 번호:' 패턴으로 마크다운을 쪼갭니다.
    sections = content.split("## Item ")
    knowledge_base = {}
    
    for section in sections:
        if not section.strip():
            continue
        # 섹션의 첫 부분에서 숫자(Item ID)를 추출합니다.
        match = re.match(r"(\d+)", section)
        if match:
            item_id = match.group(1)
            # 지식 베이스에 '## Item 숫자' 형태로 다시 조립해 저장
            knowledge_base[item_id] = "## Item " + section
            
    print(f"📦 [RAG 기동] 매뉴얼로부터 총 {len(knowledge_base)}개의 독립 문항 데이터베이스 빌드 완료.")
    return knowledge_base


def retrieve_relevant_manual_chunks(primary_item_id, knowledge_base):
    """
    1차 예측된 Primary Item ID를 바탕으로, 해당 본문 및 
    본문 내 '* **Related Item:**'에 적힌 연관 아이템 청크까지 매뉴얼에서 '동적 검색(Retrieval)'합니다.
    """
    retrieved_chunks = []
    
    # 1. 메인 아이템 청크 가져오기
    primary_chunk = knowledge_base.get(primary_item_id)
    if not primary_chunk:
        return "No manual guidelines found for this item."
        
    retrieved_chunks.append(primary_chunk)
    
    # 2. 메인 아이템 본문 안에서 Related Item ID 추출 (예: * **Related Item:** 36 -> '36')
    related_match = re.search(r"\*\s*\*\*Related Item:\*\*\s*(\d+)", primary_chunk)
    if related_match:
        related_item_id = related_match.group(1)
        print(f"🔗 [RAG 검색] 메인 문항({primary_item_id})에서 연관 문항({related_item_id}) 링크 감지!")
        
        # 3. 연관 아이템 청크도 매뉴얼 DB에서 함께 검색(Retrieval)하여 병합
        related_chunk = knowledge_base.get(related_item_id)
        if related_chunk:
            retrieved_chunks.append(related_chunk)
            print(f"✅ [RAG 성공] 문항 {primary_item_id}번과 {related_item_id}번 조각만 선별 완료.")
            
    return "\n\n".join(retrieved_chunks)


# ==========================================
# 3. Qwen2.5-VL 2-Step 파이프라인 (예측 -> RAG -> 채점)
# ==========================================
def load_qwen_model():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    return model, processor


def run_complete_rag_scoring(video_path, knowledge_base, model, processor, target_nframes=64):
    """
    치팅 없이 오직 영상만을 먼저 보고 메인 문항 ID를 예측한 뒤(1-Step),
    그 ID를 기반으로 매뉴얼 조각을 RAG 검색하여 최종 채점하는(2-Step) 완전한 RAG 시스템입니다.
    """
    vr = decord.VideoReader(video_path)
    video_duration = len(vr) / vr.get_avg_fps()

    # -----------------------------------------------------------------
    # Step 1: 영상만 보고 매뉴얼의 어떤 문항(숫자)에 해당하는지 1차 분류 (Zero-shot Classification)
    # -----------------------------------------------------------------
    print("\n🎬 [Step 1] 영상 분석 시작: 이 영상이 매뉴얼의 몇 번 Item인지 예측합니다...")
    
    # 지식 베이스에 있는 아이템 번호 목록 (예: ['36', '45', '46'])
    available_item_ids = sorted(list(knowledge_base.keys()))
    
    classification_prompt = f"""
    Observe the infant's physical movement and the surrounding setup in the video with high clinical concentration. 
    Your single mission is to classify this video sequence into the correct Bayley Item Number from this list: {available_item_ids}

    To accurately match the raw video with the correct manual item and prevent any directional or temporal confusion (e.g., mistaking forward walking for backward walking), systematically analyze the sequence using the following three clinical pillars:

    1. **Direction of the Stride & Trunk Movement (Anatomical Direction):**
    - **Determine Forward Movement:** Check if the child's center of mass (trunk/pelvis) is leaning and moving in the exact direction their chest and face are pointing. Look for forward swing phases where the swinging foot advances past the supporting foot in a forward vector.
    - **Determine Backward Movement:** Check if the child is intentionally stepping heels-first or toes-first into the blind space behind their back, with the pelvis moving posteriorly away from the direction the face/chest is pointing.
    - Do NOT be fooled by the camera angle or whether the child is moving closer to or further from the lens. Focus purely on the child's relative body-centered direction (Anatomical Egocentric Frame).

    2. **Environmental Context & Task Materials (Objects):**
    - Identify any test materials or equipment present in the video (e.g., a stepping path/line on the floor, a ball, stairs, a rail, small toys, chairs).
    - Match these visible objects with the 'Materials' or 'Item Instructions' defined in the Bayley manual. (e.g., If a stepping path is used for a forward walk challenge, cross-reference it with the relevant track item).

    3. **Execution Style & Constraints (Criteria Alignment):**
    - Notice the posture and support level: Is the action done independently, with intermittent assistance, or by holding onto a structure (like a rail or wall)?
    - Observe the primary motor mechanics: Is the child transferring weight (locomotion), applying force to an object (kicking/throwing), or maintaining a static posture (standing/balance)?

    Cross-reference your visual analysis with the candidate items in the manual. Determine the single most appropriate Item Number that governs this task.

    Output ONLY the raw number from the list (e.g., 45, 46). Do NOT include any prefix, suffix, punctuation, or conversational text.
    """
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "fps": target_nframes / video_duration},
                {"type": "text", "text": classification_prompt}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=10)
        predicted_id_text = processor.batch_decode(
            [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)], 
            skip_special_tokens=True
        )[0].strip()
    
    # 숫자만 추출하는 예외 처리
    predicted_item_id = "".join(re.findall(r"\d+", predicted_id_text))
    print(f"🎯 [Step 1 완료] Qwen이 예측한 메인 문항 번호: Item {predicted_item_id}")

    # -----------------------------------------------------------------
    # Step 2: 예측된 ID를 기반으로 매뉴얼 내용 동적 검색 (Retrieval Step)
    # -----------------------------------------------------------------
    print(f"🔍 [Step 2] RAG 시스템 가동: 매뉴얼에서 Item {predicted_item_id} 관련 조각 검색 중...")
    retrieved_md_context = retrieve_relevant_manual_chunks(predicted_item_id, knowledge_base)

    # -----------------------------------------------------------------
    # Step 3: 검색된 콤팩트 매뉴얼 정보만 들고 최종 정밀 채점 (Generation Step)
    # -----------------------------------------------------------------
    print("🧠 [Step 3] 최종 채점 시작: RAG로 추출된 슬림한 가이드라인만 참조하여 점수를 매깁니다...")
    
    scoring_prompt = f"""
    You are an expert clinical specialist scoring the Bayley Scales.
    Evaluate the video strictly against the RAG-retrieved manual sections provided below.
    
    ### [RAG-Retrieved Manual Guidelines]
    {retrieved_md_context}
    
    Please output your final decision strictly in the following JSON format:
    ```json
    {{
      "primary_item": {{
        "item_id": "Item {predicted_item_id}",
        "scoring_decision": "Score (e.g., 2, 1, or 0)",
        "clinical_justification": "Direct mapping between video and this item's criteria."
      }},
      "related_items_evaluation": [
        {{
          "item_id": "The related item ID specified in the manual context (if any)",
          "scoring_decision": "Score (2, 1, or 0)",
          "clinical_justification": "Direct mapping between video and the related item's criteria."
        }}
      ],
      "overall_behavioral_synthesis": "Continuous synthesis paragraph of the video."
    }}
    ```
    """
    
    final_messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "fps": target_nframes / video_duration},
                {"type": "text", "text": scoring_prompt}
            ]
        }
    ]
    
    final_text = processor.apply_chat_template(final_messages, tokenize=False, add_generation_prompt=True)
    final_image_inputs, final_video_inputs = process_vision_info(final_messages)
    final_inputs = processor(text=[final_text], images=final_image_inputs, videos=final_video_inputs, padding=True, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        final_generated_ids = model.generate(**final_inputs, max_new_tokens=1024)
        final_output = processor.batch_decode(
            [out_ids[len(in_ids):] for in_ids, out_ids in zip(final_inputs.input_ids, final_generated_ids)], 
            skip_special_tokens=True
        )[0]
        
    return final_output

# ==========================================
# 4. 메인 실행 제어부 (ass_path 필터링 및 실시간 배치 저장)
# ==========================================
if __name__ == "__main__":
    from utils.path_list_d03 import path_list_d03
    
    # 결과 보고서를 저장할 텍스트 파일 경로 정의
    OUTPUT_REPORT_PATH = PROJECT_ROOT / "config" / "bayley_scoring_reports_v2.txt"
    
    # 1. 메타데이터 CSV 로드 및 유효한 ass_path 동적 필터링
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"❌ 메타데이터 CSV를 찾을 수 없습니다: {CSV_PATH}")
        
    df = pd.read_csv(str(CSV_PATH))
    
    valid_video_paths = []
    print("\n🔍 [데이터 검색] ass_path 기준 실제 파일 존재 여부 전수 검사 중...")
    
    for idx, row in df.iterrows():
        # 데이터프레임의 ass_path 또는 common_path를 활용해 파일 경로 조립
        common_path = row['common_path']
        paths = path_list_d03(common_path)
        
        # split_video 폴더 내부의 mp4 리스트 추출
        video_folder = paths['split_video']
        if video_folder.exists():
            mp4_files = sorted(list(video_folder.glob("*.mp4")))
            for vid_path in mp4_files:
                # 💡 중복 등록 방지 및 유효 리스트 확보
                if vid_path not in valid_video_paths:
                    valid_video_paths.append(vid_path)

    print(f"🎯 검증 완료: 물리적으로 존재하는 총 [{len(valid_video_paths)}]개의 비디오 파일을 찾았습니다.")

    # 2. 파일명 치팅 없이, 순수 매뉴얼 데이터베이스(KB) 먼저 구축
    knowledge_base = build_manual_knowledge_base(MANUAL_PATH)
    
    # 3. 모델 및 프로세서 초기화
    model, processor = load_qwen_model()
    print("✅ 모든 인프라 및 가중치 로드 완료. 대규모 배치 분석을 시작합니다.\n")

    # 기존 결과 파일 초기화 (새로 시작할 때마다 덮어쓰기 위해 오픈)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("====================================================================\n")
        f.write("      [🎖️ Qwen2.5-VL X Local RAG 영유아 대대적 채점 자동화 레포트]      \n")
        f.write("====================================================================\n\n")

    # 4. 루프 가동 및 결과 실시간 파일 저장 (Append 모드)
    for i, video_path_obj in enumerate(valid_video_paths):
        video_path_str = str(video_path_obj)
        video_stem = video_path_obj.stem  # 확장자 제거한 순수 파일명 추출
        
        print(f"\n🚀 [{i+1}/{len(valid_video_paths)}] 현재 분석 중: {video_stem}.mp4")
        
        try:
            # [진짜 RAG] 파이프라인 구동 (분류 -> 검색 -> 채점) - 프레임 누수 방지 64 고정
            final_report = run_complete_rag_scoring(
                video_path_str, knowledge_base, model, processor, target_nframes=64
            )
            
            # 터미널 실시간 모니터링 출력
            print(f"✅ 분석 완료: {video_stem}")
            print(final_report)
            
            # 💡 [핵심] 하나의 비디오 채점이 끝날 때마다 실시간으로 파일에 기록 (메모리 단전/에러 대비)
            with open(OUTPUT_REPORT_PATH, "a", encoding="utf-8") as f:
                f.write(f"▶️ [VIDEO FILE]: {video_stem}\n")
                f.write("--------------------------------------------------------------------\n")
                f.write(f"{final_report}\n")
                f.write("====================================================================\n\n")
                
        except Exception as e:
            error_msg = f"❌ [{video_stem}] 분석 중 예외 발생: {str(e)}"
            print(error_msg)
            with open(OUTPUT_REPORT_PATH, "a", encoding="utf-8") as f:
                f.write(f"▶️ [VIDEO FILE]: {video_stem}\n")
                f.write("--------------------------------------------------------------------\n")
                f.write(f"{error_msg}\n")
                f.write("====================================================================\n\n")
            continue

    print(f"\n🎉 모든 작업이 완료되었습니다! 통합 결과는 다음 경로에 안전하게 저장되었습니다:\n👉 {OUTPUT_REPORT_PATH}")