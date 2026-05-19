import torch
import time
import os
import sys
from pathlib import Path
from transformers import AutoProcessor, AutoModelForCausalLM

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
print(f"📁 프로젝트 루트 경로: {PROJECT_ROOT}")

LOCAL_MODEL_PATH = str(PROJECT_ROOT / "my_gemma_model")

print(f"[{LOCAL_MODEL_PATH}] 로컬 모델 로딩 중...")

processor = AutoProcessor.from_pretrained(LOCAL_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    dtype=torch.bfloat16,  # 🚨 경고 메시지에 맞춰 torch_dtype 대신 dtype으로 변경
    device_map="auto"           
)
print(f"✅ 현재 모델이 작동 중인 장치: {model.device}")
print("--- 로컬 구동 준비 완료! ---")
print("\n[Gemma 4 로컬 챗] 종료하려면 'exit'를 입력하세요.")

# 🌟 핵심 1: 루프 바깥에 대화 기록을 누적할 빈 리스트를 만듭니다.
chat_history = []

while True:
    user_input = input("\n나: ")
    if user_input.lower() in ['exit', 'quit', '종료']:
        break

    # 🌟 핵심 2: 사용자의 질문을 기존 대화 기록에 '추가(append)' 합니다.
    chat_history.append({"role": "user", "content": [{"type": "text", "text": user_input}]})
    
    # 🌟 핵심 3: 방금 친 질문 하나가 아니라 '누적된 전체 대화 기록'을 모델에게 전달합니다.
    text = processor.apply_chat_template(chat_history, add_generation_prompt=True)
    # 수정된 코드 (채팅 템플릿 적용)

    # 1. 메시지 형식(List of Dictionaries)으로 구성
    messages = [
        {"role": "user", "content": text} # text 변수에 사용자의 질문이 들어있다고 가정
    ]

    # 2. 토크나이저의 apply_chat_template를 사용하여 올바른 프롬프트 형식으로 변환
    prompt = processor.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, # 텍스트 형태로 유지
        add_generation_prompt=True # 모델이 답변할 차례임을 알리는 토큰 추가
    )

    # 3. 변환된 프롬프트를 processor(또는 tokenizer)에 전달
    inputs = processor(text=prompt, return_tensors="pt").to(model.device)

    print("Gemma: \n", end="", flush=True)
    
    start_time = time.perf_counter()

    outputs = model.generate(
        **inputs, 
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    end_time = time.perf_counter()

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
    ]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(response)

    # 🌟 핵심 4: 모델이 대답한 내용도 잊지 않게 기록에 '추가'합니다.
    chat_history.append({"role": "assistant", "content": [{"type": "text", "text": response}]})

    generation_time = end_time - start_time
    generated_token_count = len(generated_ids[0])
    tps = generated_token_count / generation_time if generation_time > 0 else 0

    print(f"\n[⏱️ 답변 생성: {generation_time:.2f}초 | 🚀 속도: {tps:.2f} tokens/sec]")