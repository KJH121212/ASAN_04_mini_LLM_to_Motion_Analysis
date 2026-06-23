# 1. 베이스 이미지 설정
FROM docker.io/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# 2. 시스템 라이브러리 설치 (OpenCV 등)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /workspace

# 4. Conda 환경 변수 및 기본 쉘 설정
# 4. Conda 환경 변수 및 스레드 충돌 방지 설정
ENV PATH="/opt/conda/bin:$PATH"
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
SHELL ["/bin/bash", "--login", "-c"]

# 5. environment.yml 복사 및 패키지 설치
COPY environment.yml /tmp/environment.yml
RUN conda env update -n base -f /tmp/environment.yml && \
    conda clean -afy

# 6. 컨테이너 접속 시 자동으로 base 환경 활성화되도록 설정
RUN conda init bash && \
    echo "conda activate base" >> ~/.bashrc

# 7. 소스 코드 복사 (문법 수정됨)
COPY . /workspace

# 8. 기본 실행 명령
CMD ["/bin/bash"]