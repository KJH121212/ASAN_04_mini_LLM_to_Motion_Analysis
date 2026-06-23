# 1. 베이스 이미지 설정 (사용하시려는 PyTorch 이미지)
FROM docker.io/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# 2. 작업 디렉토리 설정
WORKDIR /workspace

# 3. 시스템 라이브러리 설치 (OpenCV, moviepy 등에 필수)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. environment.yml 복사
# 프로젝트 구조에 따라 경로를 조정하세요 (예: env/environment.yml)
COPY environment.yml /tmp/environment.yml

# 5. Conda를 사용하여 패키지 설치
# 별도의 가상환경을 만들지 않고 'base' 환경에 직접 설치하여 관리를 단순화합니다.
RUN conda env update -n base -f /tmp/environment.yml && \
    conda clean -afy

# 6. 소스 코드 복사 (선택 사항)
# 현재 폴더의 모든 내용을 컨테이너 안으로 복사
COPY . /workspace

# 7. 기본 실행 명령
CMD ["/bin/bash"]

# 1. Conda 초기화 및 환경 변수 설정
ENV PATH /opt/conda/bin:$PATH

# 2. 쉘 실행 시마다 conda activate base가 되도록 설정
RUN conda init bash && \
    echo "conda activate base" >> ~/.bashrc

# 3. [중요] 이후 모든 RUN 명령어가 base 환경에서 실행되도록 쉘 지정
SHELL ["/bin/bash", "--login", "-c"]