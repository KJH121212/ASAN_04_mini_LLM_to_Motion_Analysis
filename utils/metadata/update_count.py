import os  # 운영체제와 상호작용하여 파일 및 폴더 경로를 다루기 위한 내장 모듈입니다.
import sys  # 파이썬 인터프리터 제어 및 모듈 탐색 경로 설정을 위한 모듈입니다.
import pandas as pd  # 표 형태의 데이터(CSV)를 읽고 수정하기 위해 판다스 라이브러리를 불러옵니다.
from pathlib import Path  # 경로를 문자열이 아닌 객체로 다루어 코드를 직관적으로 만들어주는 모듈입니다.

# 프로젝트 최상위 폴더 경로 설정 (함수 외부에서 환경 설정을 한 번만 수행합니다)
PROJECT_ROOT = Path(__file__).parent.parent.parent  # 현재 파일 위치에서 부모 폴더로 3번 올라가 최상위 루트 경로를 잡습니다.
if str(PROJECT_ROOT) not in sys.path:  # 모듈 경로가 중복으로 추가되는 것을 방지하기 위한 안전장치입니다.
    sys.path.append(str(PROJECT_ROOT))  # 파이썬이 다른 폴더의 커스텀 모듈을 찾을 수 있도록 시스템 경로에 루트를 추가합니다.

print(f"📁 프로젝트 루트 경로: {PROJECT_ROOT}")  # 설정된 루트 경로를 콘솔에 출력하여 정상 등록되었는지 확인합니다.

from utils.path_list_d03 import path_list_d03  # 시스템 경로 추가 후, 유틸리티 폴더에서 경로 생성 함수를 성공적으로 불러옵니다.


# [핵심] 프레임과 JSON 개수를 세어 업데이트하는 독립적인 함수를 정의합니다.
def update_frame_and_json_counts(input_csv_path, output_csv_path):  # 입력 및 출력 CSV 경로를 인자로 받습니다.
    if not os.path.exists(input_csv_path):  # 인자로 전달받은 입력 CSV 파일이 실제로 존재하는지 가장 먼저 안전하게 검사합니다.
        print(f"오류: '{input_csv_path}' 파일을 찾을 수 없습니다.")  # 파일이 없을 경우 사용자에게 직관적인 에러 메시지를 출력합니다.
        return  # 처리할 파일이 없으므로 더 이상 진행하지 않고 함수를 안전하게 종료합니다.

    df = pd.read_csv(str(input_csv_path))  # 판다스를 사용하여 기존 CSV 파일을 읽어와 데이터프레임(df) 메모리에 적재합니다.
    print(f"🚀 총 {len(df)}개의 데이터에 대해 메타데이터(프레임/JSON 개수) 업데이트 작업을 시작합니다...")  # 작업 시작을 콘솔에 알립니다.

    for target in range(len(df)):  # 0부터 데이터프레임의 전체 행 개수만큼 반복문을 실행합니다.
        common_path = df.iloc[target]['common_path']  # 현재 순서(target)의 행에서 해당 비디오의 고유 상대 경로 값을 추출합니다.
        paths = path_list_d03(common_path)  # 추출한 고유 경로를 함수에 전달하여 프레임과 키포인트 폴더의 실제 경로 딕셔너리를 받아옵니다.

        frame_dir = Path(paths['frame'])  # 딕셔너리에서 프레임 경로 문자열을 가져와 강력한 파이썬 Path 객체로 변환합니다.
        keypoint_dir = Path(paths['keypoint'])  # 딕셔너리에서 키포인트 경로 문자열을 가져와 Path 객체로 변환합니다.

        # 제너레이터를 사용하여 하위 폴더의 모든 jpg 개수를 메모리 낭비 없이 효율적으로 카운트합니다.
        n_frames_count = sum(1 for _ in frame_dir.rglob('*.jpg')) if frame_dir.exists() else 0  # 폴더가 존재하면 파일 개수를 세고, 없으면 0을 반환합니다.

        # 제너레이터를 사용하여 하위 폴더의 모든 json 파일 개수를 효율적으로 카운트합니다.
        n_json_count = sum(1 for _ in keypoint_dir.rglob('*.json')) if keypoint_dir.exists() else 0  # 폴더가 존재하면 파일 개수를 세고, 없으면 0을 반환합니다.

        # 계산된 개수를 원본 데이터프레임(df)에 업데이트합니다. loc를 사용하여 특정 행(target)과 열을 정확히 타겟팅합니다.
        df.loc[target, 'n_frames'] = n_frames_count  # 타겟 행의 'n_frames' 컬럼 값을 방금 추출한 프레임 파일 개수로 갱신합니다.
        df.loc[target, 'n_json'] = n_json_count  # 타겟 행의 'n_json' 컬럼 값을 방금 추출한 json 파일 개수로 갱신합니다.

        print(f"✅ 업데이트 완료 [행 {target}] - 프레임: {n_frames_count}개, JSON: {n_json_count}개")  # 현재 행의 처리 결과를 출력하여 모니터링합니다.

    # 반복문이 완전히 끝난 후, 모아둔 데이터를 단 한 번만 물리적 디스크에 저장합니다.
    df.to_csv(str(output_csv_path), index=False)  # 모든 처리가 끝난 데이터프레임을 출력 경로에 새롭게 저장하며, 불필요한 인덱스 열은 제외합니다.
    print(f"🎉 성공: 데이터가 '{Path(output_csv_path).name}' 파일로 안전하게 저장되었습니다!\n")  # 전체 프로세스가 완료되었음을 사용자에게 친절하게 알립니다.


# --- 함수 사용 예시 (어디서든 아래처럼 인자만 넘겨서 호출할 수 있습니다) ---
# DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data_d03")  # 데이터 디렉토리 절대 경로를 설정합니다.
# INPUT_CSV = DATA_DIR / "metadata_v1.1.csv"  # 기존 데이터를 읽어올 입력 파일 경로를 설정합니다.
# OUTPUT_CSV = DATA_DIR / "metadata_v1.2.csv"  # 새롭게 저장할 출력 파일 경로를 설정합니다.
# 
# update_frame_and_json_counts(INPUT_CSV, OUTPUT_CSV)  # 설정한 경로들을 인자로 넣어 함수를 깔끔하게 실행합니다.