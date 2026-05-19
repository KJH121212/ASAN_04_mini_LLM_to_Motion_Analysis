import json  # JSON 형식의 텍스트 데이터를 파이썬 딕셔너리로 다루기 위해 불러옵니다.
import os  # 파일의 존재 여부 등 운영체제 파일 시스템을 확인하기 위해 불러옵니다.

def get_prompt_data(json_path, target_version):  # JSON 파일 경로와 찾고자 하는 버전을 인자로 받는 함수입니다.
    # 1. 파일 안전성 검사
    if not os.path.exists(json_path):  # 입력받은 JSON 경로가 시스템에 실제로 존재하는지 확인합니다.
        print(f"오류: '{json_path}' 파일을 찾을 수 없습니다.")  # 파일이 없을 경우 에러 메시지를 출력합니다.
        return None  # 에러 상황이므로 안전하게 None을 반환하고 함수를 마칩니다.

    # 2. JSON 파일 로드 및 파싱
    try:  # 파일 읽기 중 발생할 수 있는 에러를 방지하기 위해 예외 처리 블록을 사용합니다.
        with open(json_path, 'r', encoding='utf-8') as file:  # 텍스트 깨짐을 방지하기 위해 utf-8 인코딩으로 파일을 엽니다.
            config = json.load(file)  # JSON 파일의 모든 내용을 파이썬 딕셔너리(config)로 변환합니다.
    except json.JSONDecodeError:  # 파일의 형식이 올바른 JSON 문법이 아닐 때 실행됩니다.
        print("오류: JSON 파일 형식이 올바르지 않아 데이터를 읽을 수 없습니다.")  # 구문 오류를 사용자에게 알립니다.
        return None  # 파싱에 실패했으므로 None을 반환합니다.
    except Exception as e:  # 그 외 예상치 못한 파일 권한 등의 에러를 잡아냅니다.
        print(f"알 수 없는 파일 읽기 오류 발생: {e}")  # 구체적인 에러 내용을 출력합니다.
        return None  # 안전하게 None을 반환합니다.

    # 3. 타겟 버전 검색 및 prompt_data 추출
    for exp in config.get('experiments', []):  # 'experiments' 리스트를 가져와 항목을 하나씩 순회합니다. 키가 없으면 빈 리스트를 반환하여 에러를 막습니다.
        if exp.get('version') == target_version:  # 현재 순회 중인 실험 데이터의 버전이 우리가 찾는 타겟 버전과 일치하는지 비교합니다.
            return exp.get('prompt_data')  # 버전이 일치한다면, 우리가 그토록 원하던 'prompt_data' 딕셔너리만 즉시 반환하고 함수를 완전히 종료합니다.

    # 4. 버전을 끝내 찾지 못한 경우
    print(f"경고: 파일 내에서 '{target_version}' 버전에 해당하는 데이터를 찾을 수 없습니다.")  # 버전 오타나 누락을 알리는 경고를 띄웁니다.
    return None  # 타겟 버전을 찾지 못했으므로 최종적으로 None을 반환합니다.


