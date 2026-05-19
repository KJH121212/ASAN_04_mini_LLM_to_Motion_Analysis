import os
import sys
from pathlib import Path
import json
from dotenv import load_dotenv
from huggingface_hub import login
import whisperx

# 프로젝트 최상위 폴더 경로 설정 
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.huggingface_login import login_to_huggingface

def process_diarization(video_path: str, output_json_path: str, hf_token: str):
    """
    WhisperX를 활용하여 비디오의 음성을 추출하고 화자를 분리하여 JSON으로 저장하는 함수입니다.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"❌ 비디오 파일을 찾을 수 없습니다: {video_path}")

    # 연산 장치 설정 (NVIDIA GPU가 있다면 "cuda", 없다면 "cpu"로 설정하세요)
    device = "cpu"  
    compute_type = "int8" if device == "cpu" else "float16"

    print("\n▶️ 1. 오디오 로드 및 기본 STT 모델 구동 중...")
    audio = whisperx.load_audio(video_path)
    model = whisperx.load_model("base", device, compute_type=compute_type)
    result = model.transcribe(audio, batch_size=8)

    print("▶️ 2. 정밀 시간 동기화(Alignment) 처리 중...")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    print("▶️ 3. 화자 분리(Speaker Diarization) 분석 중... (시간이 소요됩니다)")
    # 인가받은 Hugging Face 토큰을 사용하여 화자 분리 파이프라인을 가동합니다.
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarize_segments = diarize_model(audio)
    
    # 인식된 텍스트와 분리된 화자 정보를 하나로 매핑합니다.
    result = whisperx.assign_word_speakers(diarize_segments, result)

    print("▶️ 4. 분석 결과를 JSON 파일로 저장 중...")
    subtitle_data = []
    
    for segment in result["segments"]:
        subtitle_data.append({
            "start_time": round(segment["start"], 3),
            "end_time": round(segment["end"], 3),
            "speaker": segment.get("speaker", "UNKNOWN"),  # 화자 식별 실패 시 UNKNOWN 처리
            "text": segment["text"].strip()
        })

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(subtitle_data, f, ensure_ascii=False, indent=4)
        
    print(f"🎉 모든 작업이 완료되었습니다! 결과물: {output_json_path}")


# ==========================================
# 🚀 메인 실행부 (Main Execution Block)
# ==========================================
if __name__ == "__main__":
    # 프로젝트 환경에 맞게 경로를 설정합니다.
    ENV_FILE_PATH = PROJECT_ROOT / ".env"
    TARGET_VIDEO = "../data/p13_gross_motor_1.mp4"
    OUTPUT_JSON = "../data/a.json"

    try:
        # 1. 시스템 보안 인가 및 토큰 확보
        print("🔐 시스템 초기화 및 권한 확인을 시작합니다...")
        secure_token = login_to_huggingface(env_path=ENV_FILE_PATH)

        # 2. 메인 AI 파이프라인 가동
        process_diarization(
            video_path=TARGET_VIDEO, 
            output_json_path=OUTPUT_JSON, 
            hf_token=secure_token
        )
        
    except Exception as e:
        print(f"\n❌ 프로그램 실행 중 오류가 발생했습니다:\n{e}")