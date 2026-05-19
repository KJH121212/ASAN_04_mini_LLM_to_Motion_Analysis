import os  # 운영체제와 상호작용하여 파일 및 폴더 경로를 탐색하기 위한 내장 모듈을 불러옵니다.
import pandas as pd  # 데이터프레임을 생성하고 기존 데이터와 병합/정렬하기 위해 판다스 라이브러리를 불러옵니다.

def update_video_metadata(base_path, input_csv, output_csv):  # 탐색 폴더, 입력 CSV, 출력 CSV 경로를 인자로 받는 함수를 정의합니다.
    # 이전 대화에서 개선했던 다중 동영상 확장자 지원 기능을 함수 내부 기본값으로 설정합니다.
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v')  # 탐색할 동영상 확장자들을 튜플로 정의합니다.

    # 1. 기존 지정된 input_csv 파일이 존재하면 읽어오고, 없으면 빈 데이터프레임 생성
    if os.path.exists(input_csv):  # 함수의 인자로 전달받은 input_csv 파일이 실제 시스템에 존재하는지 검사합니다.
        df_existing = pd.read_csv(input_csv)  # 기존 CSV 파일에 있던 데이터 행들을 판다스 데이터프레임으로 불러옵니다.
        print(f"기존 CSV를 성공적으로 불러왔습니다. (현재 {len(df_existing)}개 행 존재)")  # 데이터 로드 성공 및 기존 행 개수를 알립니다.
    else:  # 입력용 CSV 파일이 지정된 경로에 존재하지 않을 때 실행되는 예외 블록입니다.
        df_existing = pd.DataFrame()  # 병합 시 에러가 나지 않도록 칼럼이 텅 빈 형태의 깨끗한 데이터프레임을 초기화합니다.
        print(f"안내: 기존 CSV('{input_csv}') 파일이 없어 새로 생성 프로세스를 진행합니다.")  # 파일이 없어 새로 생성됨을 안내합니다.

    # 2. 폴더를 돌며 현재 실제 저장되어 있는 동영상 파일 목록 수집
    data_list = []  # 새로 탐색되어 수집된 파일들의 메타데이터 구조를 임시 저장할 빈 리스트를 선언합니다.

    if not os.path.exists(base_path):  # 탐색하고자 하는 비디오 기본 디렉토리가 존재하는지 가장 먼저 검사합니다.
        print(f"오류: '{base_path}' 경로를 찾을 수 없습니다. 경로를 확인해주세요.")  # 디렉토리가 없을 때 경고 메시지를 출력합니다.
        return  # 탐색할 폴더가 없으므로 더 이상 진행하지 않고 함수를 안전하게 종료합니다.

    for root, dirs, files in os.walk(base_path):  # base_path 디렉토리 내부를 재귀적으로 깊숙이 파고들며 모든 파일들을 순회합니다.
        for file in files:  # 현재 탐색 중인 폴더 디렉토리 내에 있는 파일들을 하나씩 꺼내어 반복합니다.
            if file.lower().endswith(video_extensions):  # 파일 확장자를 소문자로 안전하게 통일하여 지정된 동영상 파일만 필터링합니다.
                video_path = os.path.join(root, file)  # 현재 폴더 절대 경로와 영상 파일 이름을 결합하여 파일의 전체 경로를 만듭니다.
                rel_path = os.path.relpath(video_path, base_path)  # 전체 경로에서 base_path 구역을 제외한 순수 하위 상대 경로를 도출합니다.
                common_path = os.path.splitext(rel_path)[0]  # 상대 경로 문자열에서 뒷부분의 확장자를 분리하고 순수 이름만 추출합니다.
                
                data_list.append({  
                    "video_path": video_path,
                    "common_path": common_path,
                    "n_frames": 0,
                    "n_json": 0,
                    "frames_done": False,
                    "sapiens_done": False,
                    "reextract_done": False,
                    "overlay_done": False,
                    "is_train": False,
                    "is_val": False,
                    "id_done": False
                })

    # 3. 새 데이터프레임 생성
    df_new = pd.DataFrame(data_list)  # 이번 탐색에서 새로 수집한 딕셔너리 배열을 판다스 데이터프레임으로 빌드합니다.

    # 4. 기존 데이터 행 유지하며 신규 파일만 아래로 결합
    if df_existing.empty:  # 만약 기존 input_csv에 데이터가 아무것도 없거나 파일이 없었다면 실행됩니다.
        df_combined = df_new  # 새로 수집된 데이터프레임 전체를 최종 병합 데이터프레임으로 취급합니다.
    else:  # 기존 CSV 데이터 행들이 안전하게 저장되어 불러와진 경우 실행되는 핵심 병합 블록입니다.
        # 중복 누적을 방지하기 위해 신규 수집 데이터 중 기존 데이터의 video_path에 없는 진짜 새로운 행만 필터링합니다.
        df_new_filtered = df_new[~df_new['video_path'].isin(df_existing['video_path'])]  # 중복 행들을 깔끔하게 골라냅니다.
        df_combined = pd.concat([df_existing, df_new_filtered], ignore_index=True)  # 기존 행들을 100% 유지하면서 새 행만 밑에 이어붙입니다.

    # 5. 최종 데이터프레임 오름차순 정렬 및 output_csv 저장
    if not df_combined.empty:  # 병합이 완료된 전체 데이터프레임에 처리할 데이터 행이 존재하는지 최종 확인합니다.
        # 행 전체를 온전하게 유지한 상태에서, video_path 컬럼 기준으로 사전순(A-Z) 오름차순 정렬을 완벽하게 수행합니다.
        df_combined = df_combined.sort_values(by="video_path", ascending=True, ignore_index=True)  # 정렬 결과를 최종 변수에 안전하게 업데이트합니다.
        df_combined.to_csv(output_csv, index=False)  # 판다스 자체 행 인덱스 번호는 떼어내고, 정렬된 순수 데이터 행들만 지정된 경로로 저장합니다.
        print(f"업데이트 성공: 총 {len(df_combined)}개의 행이 오름차순 정렬되어 '{output_csv}'에 저장되었습니다.\n")  # 최종 저장 성공 메시지를 출력합니다.
    else:  # 기존 파일에도 데이터가 없고, 새로 뒤진 폴더에도 영상 파일이 전혀 없을 때의 예외 처리입니다.
        print("안내: 처리하거나 저장할 비디오 메타데이터 행이 존재하지 않습니다.\n")  # 작업 대상이 없음을 콘솔에 부드럽게 출력합니다.