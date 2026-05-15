#!/bin/bash

# 1. 시스템 의존성 설치 (OpenCV 및 영상 처리에 필수)
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# 2. Conda 환경 생성
# environment.yml 파일이 env/ 폴더 안에 있다고 가정합니다.
conda env create -f environment.yml

# 3. 환경 활성화 안내
echo "------------------------------------------------"
echo "가상환경 생성이 완료되었습니다."
echo "아래 명령어를 입력하여 환경을 활성화하세요:"
echo "conda activate gemini"
echo "------------------------------------------------"