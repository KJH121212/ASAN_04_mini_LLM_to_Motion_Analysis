import glob
import re
import os
import sys
import copy
from pathlib import Path

import torch
import cv2
import librosa
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig

# 동영상 파일에서 지정된 개수만큼 균일하게 이미지를 뽑아내는 함수 정의
def extract_video_frames(video_path, num_frames=4):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            
    cap.release()
    if not frames:
        raise ValueError(f"비디오 프레임을 추출할 수 없습니다: {video_path}")
        
    while len(frames) < num_frames:
        frames.append(frames[-1])
        
    return frames 

def extract_audio_waveform(video_path, target_sr=16000):
    try:
        # 경고 메시지를 방지하고 mp4 코덱 호환성을 높이기 위해 
        # 무겁게 전체를 읽는 대신 librosa 내부 엔진이 알아서 우회하도록 둡니다.
        waveform, sr = librosa.load(video_path, sr=target_sr, mono=True)
        return waveform
    except Exception as e:
        print(f"⚠️ 오디오 트랙을 찾을 수 없거나 추출에 실패했습니다. (무음 처리): {e}")
        # 예외 발생 시 모델이 튕기지 않도록 1초 분량의 dummy 무음 텐서 반환
        return np.zeros(target_sr, dtype=np.float32)

class Gemma4Encoder:
    def __init__(self, model_id="google/gemma-4-e2b"):
        print("📥 4비트 양자화 멀티모달 모델 로드 중...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_id, 
            quantization_config=quantization_config, 
            device_map="auto" 
        )
        self.model.eval()

        # 특수 토큰 정보 동적 취득
        self.img_token_id = self.model.config.image_token_id
        self.img_token_str = self.processor.tokenizer.convert_ids_to_tokens(self.img_token_id)
        
        if hasattr(self.model.config, "audio_token_id") and self.model.config.audio_token_id is not None:
            self.aud_token_id = self.model.config.audio_token_id
            self.aud_token_str = self.processor.tokenizer.convert_ids_to_tokens(self.aud_token_id)
        else:
            self.aud_token_id = self.processor.tokenizer.encode("<|audio|>", add_special_tokens=False)[0]
            self.aud_token_str = "<|audio|>"
            
        print(f"🔍 [시스템 검출] 이미지 토큰 ID: {self.img_token_id} ({self.img_token_str})")
        print(f"🔍 [시스템 검출] 오디오 토큰 ID: {self.aud_token_id} ({self.aud_token_str})")

    def get_embedding(self, video_path, num_frames=4):
        print(f"🎬 '{video_path}' 멀티모달 데이터 전처리 시작...")
        
        # 1. 원본 소스 전처리 함수 호출
        video_frames = extract_video_frames(video_path, num_frames)
        audio_waveform = extract_audio_waveform(video_path, target_sr=16000)
        
        # 2. 멀티모달 특수 토큰 프롬프트 조립
        img_prompt = self.img_token_str * len(video_frames)
        aud_prompt = self.aud_token_str if audio_waveform is not None else ""
        prompt = f"{img_prompt}{aud_prompt}Analyze this video and audio synchronized."
        
        # 3. 프로세서 인자 매핑 (단수형인 audio와 sampling_rate 명시)
        inputs = self.processor(
            text=prompt,
            images=video_frames,
            audio=audio_waveform, 
            sampling_rate=16000,   
            return_tensors="pt"
        )
        
        # 4. GPU 메모리로 이동
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        print("🧠 멀티모달 (비디오+오디오) 통합 인코딩 진행 중...")
        
        with torch.no_grad(): 
            # 5. 모델 포워드 (모든 멀티모달 인자가 언패킹되어 주입됨)
            outputs = self.model(
                **inputs, 
                output_hidden_states=True 
            )
            
            # 최종 레이어 은닉 상태 추출 [Batch=1, Sequence_Length, Hidden_Dim]
            last_hidden_state = outputs.hidden_states[-1] 
            
            # 💡 [정밀 타격] 텍스트를 배제하고 오직 이미지와 오디오 토큰 위치만 마스킹합니다.
            input_ids = inputs['input_ids'].squeeze(0) # 1차원으로 압축
            
            # 현재 토큰 ID가 이미지 토큰이거나 오디오 토큰인 구간만 True로 마킹
            media_mask = (input_ids == self.img_token_id) | (input_ids == self.aud_token_id)
            
            # 만약 특수 상황 때문에 마스크가 한 군데도 안 잡히면 안전빵으로 전체 평균 적용
            if not media_mask.any():
                print("⚠️ [경고] 미디어 특수 토큰 인덱스를 잡지 못했습니다. 전체 평균으로 대체합니다.")
                media_embedding = last_hidden_state.mean(dim=1).squeeze()
            else:
                # 미디어 토큰이 존재하는 차원 영역만 필터링하여 평균(Mean Pooling) 계산
                # last_hidden_state.squeeze(0) 결과 크기: [Sequence_Length, Hidden_Dim]
                media_embedding = last_hidden_state.squeeze(0)[media_mask].mean(dim=0)
            
            # 6. NumPy 배열로 형변환하여 CPU로 반환
            final_embedding = media_embedding.to(torch.float32).cpu().numpy()
            
        print(f"✅ 멀티모달 임베딩 추출 성공! 벡터 차원 크기: {final_embedding.shape}")
        return final_embedding

# if __name__ == "__main__":
#     torch.cuda.empty_cache()
    
#     # 인코더 기동
#     encoder = Gemma4Encoder("google/gemma-4-e2b")
#     target_video = "../data/p13_gross_motor_1.mp4" 
    
#     # 멀티모달 임베딩 추출
#     vector_result = encoder.get_embedding(target_video)
#     print("🎬🔊 융합 임베딩 앞 5개 숫자:", vector_result[:5])

if __name__ == "__main__":
    # 1. 환경 및 파일 준비
    import torch
    torch.cuda.empty_cache()
    
    target_video = "../data/p13_gross_motor_1.mp4" 
    
    # 2. 인코더 객체 생성 (토큰 및 모델 아키텍처 로드)
    encoder = Gemma4Encoder("google/gemma-4-e2b")
    
    print("\n" + "="*70)
    print("🔍 [인자 추적] 1. 원본 미디어 데이터 추출")
    print("="*70)
    video_frames = extract_video_frames(target_video, num_frames=4)
    audio_waveform = extract_audio_waveform(target_video, target_sr=16000)
    
    print(f"   - 추출된 이미지 장수: {len(video_frames)}장")
    print(f"   - 추출된 오디오 배열 크기: {audio_waveform.shape}")

    print("\n" + "="*70)
    print("🔍 [인자 추적] 2. self.processor 출력 인자 및 텐서 스펙 확인")
    print("="*70)
    
    # 가상의 통합 프롬프트 구성
    img_prompt = encoder.img_token_str * len(video_frames)
    aud_prompt = encoder.aud_token_str if audio_waveform is not None else ""
    prompt = f"{img_prompt}{aud_prompt}Analyze this video and audio synchronized."

    # 💡 프로세서 통과
    inputs = encoder.processor(
        text=prompt,
        images=video_frames,
        audio=audio_waveform, 
        sampling_rate=16000,   
        return_tensors="pt"
    )
    
    # 🎯 딕셔너리 내부의 모든 인자(Key)와 텐서 모양(Shape)을 동적으로 순회하며 출력합니다.
    print(f"💡 self.processor가 생성한 총 인자 개수: {len(inputs)}개\n")
    
    for key, value in inputs.items():
        if hasattr(value, "shape"):
            print(f"   ▶️ 인자명: '{key}'")
            print(f"      - 데이터 타입: {type(value)}")
            print(f"      - 텐서 차원 (Shape): {list(value.shape)}")
            print(f"      - 하드웨어 디바이스: {value.device}\n")
        else:
            # 혹시 텐서가 아닌 일반 리스트나 정수형태가 있다면 예외 출력
            print(f"   ▶️ 인자명: '{key}' (Non-Tensor)")
            print(f"      - 데이터 값/타입: {type(value)}\n")
            
    print("="*70)
