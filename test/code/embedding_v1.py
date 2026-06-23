# 파이토치: 딥러닝 모델을 텐서(Tensor) 연산으로 구동하기 위한 핵심 라이브러리
import torch
# OpenCV: 비디오 파일에서 영상 프레임(이미지)을 추출하기 위한 컴퓨터 비전 라이브러리
import cv2
# Librosa: 오디오 파일을 로드하고 주파수(Sample Rate) 등을 변환하는 음향 처리 라이브러리
import librosa
# Numpy: 다차원 배열 및 수학적 계산을 위한 라이브러리 (최종 벡터 값을 담는 데 사용)
import numpy as np
# PIL (Pillow): 파이썬에서 이미지를 메모리 상의 객체로 다루기 위한 라이브러리
from PIL import Image
# Transformers: 허깅페이스(Hugging Face)에서 제공하는 사전 학습된 모델과 프로세서를 불러오는 라이브러리
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig

# 동영상 파일에서 지정된 개수만큼 균일하게 이미지를 뽑아내는 함수 정의
def extract_video_frames(video_path, num_frames=4):
    # cv2.VideoCapture를 사용해 비디오 파일을 엽니다.
    cap = cv2.VideoCapture(video_path)
    # 비디오의 전체 프레임 수(총 사진 장수)를 정수형으로 가져옵니다.
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 0부터 전체 프레임 수 사이에서, 우리가 원하는 장수(num_frames)만큼 균일한 간격의 인덱스 번호를 계산합니다.
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    # 추출한 프레임(이미지)들을 담을 빈 리스트를 생성합니다.
    frames = []
    # 전체 비디오 프레임을 처음부터 끝까지 순회합니다.
    for i in range(total_frames):
        # cap.read()로 프레임을 한 장씩 읽어옵니다. ret는 성공 여부, frame은 이미지 데이터입니다.
        ret, frame = cap.read()
        # 만약 비디오가 끝났거나 에러가 나서 프레임을 못 읽었다면 반복문을 탈출합니다.
        if not ret:
            break
        # 현재 읽은 프레임 번호(i)가 우리가 추출하기로 한 인덱스 배열(indices) 안에 있다면,
        if i in indices:
            # OpenCV는 색상을 BGR 순서로 읽기 때문에, 이를 일반적인 RGB 순서로 변환해 줍니다.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 변환된 RGB 배열을 PIL 이미지 객체로 만들어 리스트에 추가합니다.
            frames.append(Image.fromarray(frame_rgb))
            
    # 비디오 파일의 점유를 해제(닫기)합니다. 메모리 누수를 방지합니다.
    cap.release()
    # (예외 처리) 만약 영상이 너무 짧아서 목표한 프레임 수(4장)를 다 못 채웠을 경우,
    while len(frames) < num_frames:
        # 마지막으로 추출된 프레임을 계속 복사해서 넣어 목표 장수를 강제로 맞춥니다.
        frames.append(frames[-1])
        
    # 최종적으로 PIL 이미지 객체들이 담긴 리스트를 반환합니다.
    return frames 

# (임시 비활성화됨) 비디오에서 오디오 파형을 추출하는 함수 정의
def extract_audio_waveform(video_path, target_sr=16000):
    try:
        # librosa를 사용해 비디오 파일에서 소리만 추출하고, 주파수를 16kHz 단일 채널(mono)로 변환해 로드합니다.
        waveform, sr = librosa.load(video_path, sr=target_sr, mono=True)
        # 성공적으로 로드된 1차원 오디오 파형 배열을 반환합니다.
        return waveform
    except Exception as e:
        # 오디오 트랙이 없거나 코덱 문제로 실패한 경우 에러 메시지를 출력합니다.
        print(f"오디오 추출 실패: {e}")
        # 에러가 나면 프로그램이 뻗지 않도록 5초 분량의 무음(0으로 가득 찬 배열)을 대신 반환합니다.
        return np.zeros(target_sr * 5)

# 모델 로드 및 임베딩 추출을 담당하는 클래스 정의
class Gemma4Encoder:
    # 클래스 초기화 시 실행되는 함수. 기본 모델은 가벼운 Gemma-4-e2b를 사용합니다.
    def __init__(self, model_id="google/gemma-4-e2b"):
        # 초기화 시작을 알리는 안내 메시지 출력
        print("📥 4비트 양자화 모델 로드 중...")
        # VRAM 절약을 위해 모델을 4비트(NF4) 형식으로 압축해서 불러오는 설정을 만듭니다.
        quantization_config = BitsAndBytesConfig(
            # 4비트 로드 기능을 활성화합니다 (VRAM 요구량이 1/4로 줄어듦).
            load_in_4bit=True,
            # 연산할 때의 데이터 타입은 최신 GPU에 유리한 bfloat16을 사용하도록 설정합니다.
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        # 텍스트, 이미지, 오디오 데이터를 모델이 이해할 수 있는 텐서로 바꿔주는 '프로세서'를 다운/로드합니다.
        self.processor = AutoProcessor.from_pretrained(model_id)
        # 실제 모델(뇌)을 양자화 설정과 함께 다운/로드합니다.
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_id, # 모델 이름
            quantization_config=quantization_config, # 4비트 압축 설정 적용
            device_map="auto" # 모델 파트를 VRAM 상황에 맞춰 GPU에 자동으로 할당합니다.
        )
        # 모델을 학습(Training) 모드가 아닌 평가/추론(Evaluation) 모드로 전환하여 불필요한 메모리 사용을 막습니다.
        self.model.eval()

        # 모델 설정(config) 파일에서 이미지 자리를 뜻하는 특수 토큰의 고유 번호(ID)를 가져옵니다.
        self.img_token_id = self.model.config.image_token_id
        # 토크나이저를 사용해 그 번호(ID)를 실제 문자열(예: <|image|>)로 변환하여 변수에 저장합니다.
        self.img_token_str = self.processor.tokenizer.convert_ids_to_tokens(self.img_token_id)
        # 하드코딩으로 인한 에러를 막기 위해 동적으로 찾은 진짜 이미지 토큰을 출력해 확인합니다.
        print(f"🔍 모델 내부에서 감지된 [진짜 이미지 토큰]: {self.img_token_str}")
        
        # 오디오 토큰 문자열을 담을 빈 변수를 만듭니다.
        self.aud_token_str = ""
        # 현재 불러온 모델의 설정(config)에 오디오 토큰 번호(audio_token_id)가 존재하는지 확인합니다.
        if hasattr(self.model.config, "audio_token_id"):
            # 존재한다면 해당 번호를 실제 문자열(예: <|audio|>)로 변환해 저장합니다.
            self.aud_token_str = self.processor.tokenizer.convert_ids_to_tokens(self.model.config.audio_token_id)
            # 동적으로 찾은 오디오 토큰을 출력해 확인합니다.
            print(f"🔍 모델 내부에서 감지된 [진짜 오디오 토큰]: {self.aud_token_str}")

    # 실제 비디오 경로를 받아 벡터(임베딩)를 추출해 내는 메인 함수
    def get_embedding(self, video_path, num_frames=4):
        # 작업 시작을 알리는 안내 메시지 출력
        print(f"🎬 '{video_path}' 데이터 전처리 중...")
        
        # 앞서 정의한 함수를 호출해 비디오 파일에서 4장의 프레임(이미지)을 추출합니다.
        video_frames = extract_video_frames(video_path, num_frames)
        

        # 오디오 파형 추출 코드를 주석 처리하여 현재 실행되지 않도록 막아두었습니다.
        audio_waveform = extract_audio_waveform(video_path) # <- 주석 처리 또는 삭제
        
        # 추출한 이미지 장수(4장)만큼 모델의 진짜 이미지 토큰(예: <|image|>)을 곱해서 텍스트로 만듭니다.
        img_prompt = self.img_token_str * len(video_frames)
        
        # 오디오가 비활성화되었으므로 프롬프트에 이미지 토큰만 넣고 명령어를 덧붙입니다.
        prompt = f"{img_prompt}Analyze this video."
        
        # 프로세서를 사용해 프롬프트(텍스트)와 이미지 프레임을 파이토치 텐서("pt") 형태로 변환합니다.
        # 오디오 파라미터는 제외되었습니다.
        inputs = self.processor(
            text=prompt, # 완성된 특수 텍스트 프롬프트
            images=video_frames, # 추출된 4장의 PIL 이미지 리스트
            return_tensors="pt" # 결과를 파이토치(PyTorch) 텐서 포맷으로 반환하라는 옵션
        )
        
        # 텐서로 변환된 입력 데이터들(input_ids, pixel_values 등)을 모델이 올라가 있는 GPU 메모리로 이동시킵니다.
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # 모델 추론 시작을 알리는 메시지 출력
        print("🧠 E2B 모델 인코딩 중... (텍스트 생성 기능 OFF)")
        
        # 기울기(Gradient) 계산을 비활성화하여 VRAM을 크게 절약하고 연산 속도를 높입니다. (추론 시 필수)
        with torch.no_grad(): 
            # 텍스트를 한 글자씩 생성(generate)하는 대신, 모델을 한 번만 통과(forward)시킵니다.
            outputs = self.model(
                **inputs, # 딕셔너리 형태의 입력 텐서들을 언패킹하여 모델에 전달
                output_hidden_states=True # 텍스트 결과 대신 모델 내부의 중간 뇌파(Hidden states)를 반환하라는 핵심 옵션!
            )
            
            # 모델의 여러 층(Layers) 중에서 가장 마지막 층에서 나온 최종 은닉 상태(Hidden state)를 가져옵니다.
            last_hidden_state = outputs.hidden_states[-1] 
            
            # 여러 토큰들의 벡터값을 하나로 평균(mean) 내고, 불필요한 껍데기 차원(Batch=1 등)을 없앱니다(squeeze).
            # Numpy에서 BFloat16을 지원하지 않으므로, 일반 float32로 변환 후 GPU(VRAM)에서 CPU(시스템 메모리)로 내리고 Numpy 배열로 최종 변환합니다.
            video_embedding = last_hidden_state.mean(dim=1).squeeze().to(torch.float32).cpu().numpy()    
            
        # 추출이 완료되었음을 알리고, 추출된 1차원 배열(벡터)의 크기를 출력합니다.                    
        print(f"✅ 임베딩 추출 완료! 벡터 차원 크기: {video_embedding.shape}")
        # 최종적으로 완성된 1차원 Numpy 배열(벡터)을 반환합니다.
        return video_embedding

# 이 파이썬 파일이 모듈로 불리지 않고, 직접 실행되었을 때만 작동하는 메인 블록
if __name__ == "__main__":
    # 혹시 GPU 메모리에 이전 작업의 찌꺼기가 남아있을 수 있으니 캐시를 깨끗하게 한 번 비워줍니다.
    torch.cuda.empty_cache()
    # 위에서 정의한 Gemma4Encoder 클래스의 객체를 생성합니다. (이때 모델이 VRAM에 로드됩니다.)
    encoder = Gemma4Encoder("google/gemma-4-e2b")
    # 분석할 테스트 비디오 파일의 경로를 지정합니다.
    target_video = "../data/p13_gross_motor_1.mp4" 
    
    # 생성된 인코더 객체의 get_embedding 함수를 호출하여 비디오의 벡터(숫자 배열)를 추출합니다.
    vector_result = encoder.get_embedding(target_video)
    # 1536개의 숫자 중 값이 제대로 들어갔는지 확인하기 위해 맨 앞 5개의 숫자를 잘라서 콘솔에 출력합니다.
    print("추출된 임베딩 앞 5개 숫자:", vector_result[:5])