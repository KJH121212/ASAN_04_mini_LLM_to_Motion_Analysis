import glob
import re
import os
import sys
import copy
from pathlib import Path
import pandas as pd

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

# 비디오에서 오디오 파형을 추출하는 함수 정의 (주석 해제 및 안정화)
def extract_audio_waveform(video_path, target_sr=16000):
    try:
        # librosa를 사용해 비디오 파일에서 소리만 추출하고, 주파수를 16kHz 단일 채널(mono)로 변환해 로드합니다.
        waveform, sr = librosa.load(video_path, sr=target_sr, mono=True)
        return waveform
    except Exception as e:
        # 오디오 트랙이 없거나 코덱 문제로 실패한 경우 에러 메시지를 출력합니다.
        print(f"⚠️ 오디오 추출 실패(무음 대체): {e}")
        # 에러가 나면 프로그램이 뻗지 않도록 5초 분량의 무음(0으로 가득 찬 배열)을 대신 반환합니다.
        return np.zeros(target_sr * 5)

# 모델 로드 및 비디오+오디오 통합 임베딩 추출을 담당하는 클래스 정의
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

        # [특수 토큰 매핑] 모델 내부에서 이미지 특수 토큰 확보
        self.img_token_id = self.model.config.image_token_id
        self.img_token_str = self.processor.tokenizer.convert_ids_to_tokens(self.img_token_id)
        print(f"🔍 [진짜 이미지 토큰]: {self.img_token_str}")
        
        # [특수 토큰 매핑] 모델 내부에서 오디오 특수 토큰 확보
        self.aud_token_str = ""
        if hasattr(self.model.config, "audio_token_id") and self.model.config.audio_token_id is not None:
            self.aud_token_str = self.processor.tokenizer.convert_ids_to_tokens(self.model.config.audio_token_id)
            print(f"🔍 [진짜 오디오 토큰]: {self.aud_token_str}")
        else:
            # 하드코딩 예외 방어선 설정
            self.aud_token_str = "<|audio|>"
            print(f"🔍 [기본 오디오 토큰 대체]: {self.aud_token_str}")

    # 실제 비디오 경로를 받아 비디오와 오디오를 융합한 벡터(임베딩)를 추출해 내는 메인 함수
    def get_embedding(self, video_path, num_frames=4):
        print(f"🎬 '{video_path}' 멀티모달 데이터 전처리 시작...")
        
        # 1. 멀티모달 소스 데이터 추출 (비디오 프레임 + 오디오 웨이브폼)
        video_frames = extract_video_frames(video_path, num_frames)
        audio_waveform = extract_audio_waveform(video_path, target_sr=16000)
        
        # 2. 멀티모달 통합 특수 프롬프트 생성
        # 추출한 이미지 장수(4장)만큼 이미지 토큰을 복사합니다. (예: <|image|><|image|><|image|><|image|>)
        img_prompt = self.img_token_str * len(video_frames)
        
        # 💡 [핵심] 오디오 토큰 주입 (오디오 입력 데이터가 전달될 것임을 프롬프트에 명시)
        # MMEngine/Huggingface 멀티모달 포맷에 따라 오디오 토큰 1개를 조합합니다.
        aud_prompt = self.aud_token_str if audio_waveform is not None else ""
        
        # 최종 통합 명령어 구성
        prompt = f"{img_prompt}{aud_prompt}Analyze this video and audio synchronized."
        
        # 3. 프로세서를 통한 텐서 데이터 정렬
        # 텍스트, 이미지 리스트, 오디오 파형 배열을 통째로 넘겨 모델 전용 딕셔너리를 생성합니다.
        inputs = self.processor(
            text=prompt,
            images=video_frames,
            audios=audio_waveform, # 💡 오디오 데이터 파라미터 공식 부활
            sampling_rate=16000,   # 💡 오디오 샘플링 레이트 동기화 명시
            return_tensors="pt"
        )
        
        # 4. 연산 가속을 위해 GPU 메모리로 이동
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        print("🧠 멀티모달 (비디오+오디오) 통합 인코딩 진행 중...")
        
        # 5. 순방향 패스(Forward)를 통한 임베딩 추출
        with torch.no_grad(): 
            outputs = self.model(
                **inputs, 
                output_hidden_states=True # 중간 숨은 레이어 차원 반환 활성화
            )
            
            # 모델 최종 레이어의 뇌파(Hidden State) 취득
            last_hidden_state = outputs.hidden_states[-1] 
            
            # 전체 토큰 축(dim=1)을 평균(Mean Pooling)하여 멀티모달 통합 시퀀스 벡터 1개로 압축
            video_embedding = last_hidden_state.mean(dim=1).squeeze().to(torch.float32).cpu().numpy()    
            
        print(f"✅ 멀티모달 임베딩 추출 성공! 벡터 차원 크기: {video_embedding.shape}")
        return video_embedding

if __name__ == "__main__":
    torch.cuda.empty_cache()
    
    csv_path = "../metadata_v1.0.csv"
    df = pd.read_csv(csv_path)

    target = 0
    common_path = df.iloc[0]['common_path']

    ASS_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/d03/ass")
    ASS_PATH = ASS_DIR / f"{common_path}.ass"
    print(f"{ASS_PATH}")

    # # 인코더 기동
    # encoder = Gemma4Encoder("google/gemma-4-e2b")
    # target_video = "../data/p13_gross_motor_1.mp4" 
    
    # # 멀티모달 임베딩 추출
    # vector_result = encoder.get_embedding(target_video)
    # print("🎬🔊 융합 임베딩 앞 5개 숫자:", vector_result[:5])