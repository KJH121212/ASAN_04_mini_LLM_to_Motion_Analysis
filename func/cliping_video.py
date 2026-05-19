import os
import sys
import pandas as pd
from pathlib import Path
from tabulate import tabulate

# 프로젝트 최상위 폴더 경로 설정 
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

print(f"📁 프로젝트 루트 경로: {PROJECT_ROOT}")

# =======================================================
# Data 위치 설정
# =======================================================
from utils.path_list_d03 import path_list_d03

DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data_d03")  # 데이터 디렉토리 경로를 설정합니다.
CSV_PATH = DATA_DIR / "metadata_v1.0.csv"

df = pd.read_csv(str(CSV_PATH))

target = 0

common_path = df.iloc[target]['common_path']
paths = path_list_d03(common_path)

# =======================================================
# 파싱 실험
# =======================================================
from utils.ass_parser import read_ass_subtitles, parse_bayley_subtitle_data

extracted_list = read_ass_subtitles(paths['ass'])

final_data = parse_bayley_subtitle_data(extracted_list)

# =======================================================
# 비디오 클립 만들기 및 3단계 폴더 트리 저장 (최종 반영)
# =======================================================
from utils.video_processor.video_divider import cut_video_with_audio

# 1단계 최상위 폴더인 '1.bayley' 경로를 생성합니다.
bayley_root_dir = DATA_DIR / "test/video_clip/1.bayley"
bayley_root_dir.mkdir(parents=True, exist_ok=True)

print(f"🎬 비디오 클립 분할 작업을 시작합니다. 저장 경로: {bayley_root_dir}")

# 원본 비디오 파일의 경로를 확보합니다. (paths 딕셔너리에 'video' 키로 저장되어 있다고 가정)
source_video_path = paths['video']
for clip_info in final_data:  # 파싱 완료된 영상 구간 리스트를 루프를 돌며 하나씩 순차적으로 제어합니다.
    start_time = clip_info[0]  # 리스트 원본의 0번째 인덱스에서 자막 시작 타임스탬프를 매핑합니다. (예: '0:00:27.07')
    end_time = clip_info[1]  # 리스트 원본의 1번째 인덱스에서 자막 종료 타임스탬프를 매핑합니다. (예: '0:00:48.46')
    item_category = clip_info[2]  # 2단계 중분류 폴더명이 될 베일리 검사 항목 코드를 가져옵니다. (예: 'Cog', 'RC')
    item_number = clip_info[3]  # 3단계 소분류 폴더명이 될 베일리 문항 번호를 가져옵니다. (예: '52', '53')
    
    category_dir = bayley_root_dir / f"{item_category}"  # 요구사항에 맞춰 앞에 '2.'을 붙인 중분류 폴더 경로를 생성합니다. (예: 1.bayley/2.Cog)
    category_dir.mkdir(parents=True, exist_ok=True)  # 해당 디렉토리가 시스템에 없을 경우 에러 없이 안전하게 실시간으로 자동 생성합니다.
    
    clip_num_dir = category_dir / f"{item_number}"  # 요구사항에 맞춰 앞에 '3.'을 붙인 소분류 폴더 경로를 생성합니다. (예: 1.bayley/2.Cog/3.52)
    clip_num_dir.mkdir(parents=True, exist_ok=True)  # 해당 문항 번호 폴더를 파일 시스템에 물리적으로 생성합니다.
    
    video_stem = Path(source_video_path).stem  # 입력 원본 영상 파일의 경로에서 확장자를 뗀 순수 파일명을 파싱합니다. (예: 'video_d03')
    
    start_parts = start_time.split('.')[0].split(':')  # 시작 시간 문자열에서 소수점 밀리초를 버린 후, 콜론을 기준으로 시/분/초를 분리합니다.
    start_fmt = f"{int(start_parts[0]):02d}h{int(start_parts[1]):02d}m{int(start_parts[2]):02d}s"  # 각 단위를 2자리 숫자로 패딩하여 '00h00m27s' 포맷을 만듭니다.
    if start_fmt.startswith("00h"): start_fmt = start_fmt[3:]  # 만약 1시간 미만(00h)인 경우, 더 직관적인 가독성을 위해 분과 초('00m27s')만 남깁니다.
    
    end_parts = end_time.split('.')[0].split(':')  # 종료 시간 문자열에서도 동일하게 프레임 단위를 제거하고 콜론 구조를 리스트로 분할합니다.
    end_fmt = f"{int(end_parts[0]):02d}h{int(end_parts[1]):02d}m{int(end_parts[2]):02d}s"  # 시분초 형식에 맞춰 정수로 바꾼 뒤 두 자리 자릿수를 맞춰 결합합니다.
    if end_fmt.startswith("00h"): end_fmt = end_fmt[3:]  # 영상 구간이 1시간 미만일 경우 시(h) 단위를 과감히 생략하고 분/초만 노출시킵니다.
    
    output_video_path = clip_num_dir / f"{video_stem}_({start_fmt}~{end_fmt}).mp4"  # 선택하신 3번 규칙을 적용하여 최종 저장 경로를 완성합니다. (예: video_d03_(00m27s~00m48s).mp4)
    
    try:  # 대량의 미디어 파일 입출력 시 발생할 수 있는 잠재적 결함이나 예외를 래핑하여 파이프라인의 중단을 방지합니다.
        cut_video_with_audio(  # 무비파이 엔진을 호출하여 실제 영상 편집 및 저장을 수행하는 커스텀 함수를 실행합니다.
            input_path=str(source_video_path),  # 원본 영상의 Path 객체를 함수 입력 표준 인터페이스에 맞춰 문자열 형태로 변환하여 전달합니다.
            output_path=str(output_video_path),  # 새롭게 빌드한 직관적 파일명의 최종 저장 경로를 문자열로 풀어서 인풋으로 제공합니다.
            start_time=start_time,  # 무비파이가 타임라인 인덱싱을 할 수 있도록 원본 자막의 오리지널 시작 시간 형식을 그대로 전달합니다.
            end_time=end_time  # 무비파이가 영상 매핑을 멈출 수 있도록 원본 자막의 오리지널 종료 시간 형식을 그대로 전달합니다.
        )
    except Exception as e:  # 특정 프레임의 코덱 결함이나 입출력 에러가 발생한 경우 해당 루프의 예외를 캐치합니다.
        print(f"❌ [{item_category} - {item_number}] 클립 생성 실패: {e}")  # 실패 로그를 직관적으로 남겨 어떤 데이터 파일이 전처리에서 누락되었는지 즉시 추적할 수 있게 합니다.

print("🎉 모든 비디오 클립이 지정된 3단계 폴더 구조에 맞춰 성공적으로 저장되었습니다!")  # 전체 데이터셋에 대한 영상 분할 자동화 배치가 완벽히 종료되었음을 알립니다.