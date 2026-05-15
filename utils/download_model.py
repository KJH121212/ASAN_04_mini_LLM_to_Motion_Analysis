import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.huggingface_login import login_to_huggingface
login_to_huggingface(str(PROJECT_ROOT / ".env"))

def download_gemma_model():
    # 다운로드할 모델 이름과 저장할 폴더 경로 설정
    model_id = "google/gemma-4-E2B-it"
    target_dir = PROJECT_ROOT / "my_gemma_model"
    
    print(f"🚀 [{model_id}] 모델 다운로드를 시작합니다...")
    print(f"📁 저장 위치: {target_dir}")
    print("⏳ 파일 용량이 커서 시간이 오래 걸릴 수 있습니다. 창을 끄지 마세요!")

    # snapshot_download를 사용해 모델의 모든 파일을 지정한 폴더로 다운로드
    snapshot_download(
        repo_id=model_id,
        local_dir=target_dir,
    )
    
    print("\n✅ 모델 다운로드가 완벽하게 끝났습니다!")

if __name__ == "__main__":
    download_gemma_model()