import torch
import time
import os
import sys
import json
from pathlib import Path
from transformers import AutoProcessor, AutoModelForCausalLM

# 1. 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
print(f"📁 프로젝트 루트 경로: {PROJECT_ROOT}")

# ==========================================
# 🌟 핵심 1: prompt.json 파일 불러오기
# ==========================================
# (주의: prompt.json 파일이 configs 폴더 안에 있다고 가정합니다. 위치에 맞게 수정하세요)
CONFIG_PATH = PROJECT_ROOT / "configs" / "prompt.json"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    prompt_config = json.load(f)

# 최신 버전(v1.0)의 실험 설정 가져오기
target_version = prompt_config.get("latest_version", "v1.0")
current_experiment = next((exp for exp in prompt_config["experiments"] if exp["version"] == target_version), None)

if current_experiment is None:
    raise ValueError(f"❌ {target_version} 버전의 프롬프트를 찾을 수 없습니다!")

# 프롬프트 내용 및 모델 세팅값 추출
prompt_data = current_experiment["prompt_data"]
model_settings = current_experiment["model_settings"]

system_instruction = prompt_data["system_instruction"]
raw_user_message = prompt_data["user_message"]

# ==========================================
# 2. 로컬 모델 로딩
# ==========================================
LOCAL_MODEL_PATH = str(PROJECT_ROOT / "my_gemma_model")

print(f"[{LOCAL_MODEL_PATH}] 로컬 모델 로딩 중...")

processor = AutoProcessor.from_pretrained(LOCAL_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto"           
)
print(f"✅ 현재 모델이 작동 중인 장치: {model.device}")
print("--- 로컬 구동 준비 완료! ---\n")

# ==========================================
# 🌟 핵심 2: 데이터 주입 및 프롬프트 완성
# ==========================================
# 테스트용 가짜 데이터 (나중에는 실제 추출된 프롬임/좌표/텍스트 정보가 들어갑니다)
TARGET_VIDEO_DATA = "[00:10 Assessor shows red ring, 00:25 Infant touches ring]"

# JSON에서 가져온 문자열의 {video_data} 부분에 실제 데이터를 주입
final_user_message = raw_user_message.format(video_data=TARGET_VIDEO_DATA)

# Gemma 포맷에 맞게 메시지 구성 (System + User 합치기)
messages = [
    {"role": "user", "content": [{"type": "text", "text": system_instruction + "\n\n" + final_user_message}]}
]

print(f"📝 적용된 프롬프트 버전: {target_version}")
print(f"🌡️ 적용된 온도(Temperature): {model_settings['temperature']}")
print("🤖 Gemma 모델 추론 중 (JSON 포맷 출력 대기)...\n")

# ==========================================
# 3. 모델 추론 실행
# ==========================================
text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=[text], return_tensors="pt").to(model.device)

start_time = time.perf_counter()

outputs = model.generate(
    **inputs, 
    max_new_tokens=1024, # JSON이 길어질 수 있으므로 여유 있게 설정
    do_sample=True,
    temperature=model_settings["temperature"], # JSON에서 가져온 0.0 적용
    top_p=model_settings["top_p"]              # JSON에서 가져온 0.9 적용
)

end_time = time.perf_counter()

# 결과 텍스트 디코딩
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
]
response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("==================== 분석 결과 ====================")
print(response)
print("===================================================")

generation_time = end_time - start_time
generated_token_count = len(generated_ids[0])
tps = generated_token_count / generation_time if generation_time > 0 else 0

print(f"\n[⏱️ 답변 생성: {generation_time:.2f}초 | 🚀 속도: {tps:.2f} tokens/sec]")