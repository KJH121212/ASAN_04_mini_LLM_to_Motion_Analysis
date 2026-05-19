import os  # 운영체제와 상호작용하여 파일 및 폴더 경로를 다루고 이름을 변경하기 위한 내장 모듈을 불러옵니다.

def normalize_filenames(folder_path):  # 폴더 경로를 입력받아 내부 파일명들을 정규화(공백 제거, 소문자화)하는 함수를 정의합니다.
    if not os.path.exists(folder_path):  # 입력받은 폴더 경로가 실제로 시스템에 존재하는지 가장 먼저 안전하게 검사합니다.
        print(f"오류: '{folder_path}' 경로를 찾을 수 없습니다.")  # 경로가 없을 경우 사용자에게 직관적인 오류 메시지를 출력합니다.
        return  # 파일명 변경 작업을 수행하지 않고 함수를 즉시 안전하게 종료합니다.

    print(f"'{folder_path}' 내부의 파일명 정규화 작업을 시작합니다...")  # 작업이 본격적으로 시작되었음을 터미널(콘솔)에 알립니다.

    for root, dirs, files in os.walk(folder_path):  # 지정된 폴더 및 그 하위의 모든 디렉토리를 재귀적으로 깊숙이 탐색합니다.
        for file in files:  # 현재 탐색 중인 폴더(root 변수) 안에 있는 파일들을 하나씩 꺼내어 반복문을 수행합니다.
            
            new_file_name = file.replace(" ", "_").lower()  # 파일명 내부의 모든 공백을 '_'로 통일하고 전체 문자열을 소문자로 변환합니다.
            
            if new_file_name != file:  # 기존 파일명과 변환된 새 파일명이 달라, 실제로 변경 작업이 필요한 경우에만 아래 로직을 실행합니다.
                old_file_path = os.path.join(root, file)  # 이름 변경 전 기존 파일의 전체 절대 경로를 안전하게 생성합니다.
                new_file_path = os.path.join(root, new_file_name)  # 변경 후 적용될 새로운 파일의 전체 절대 경로를 생성합니다.
                
                if not os.path.exists(new_file_path):  # 변경하려는 새 이름과 동일한 파일이 폴더 내에 이미 존재하는지 중복 검사를 합니다.
                    os.rename(old_file_path, new_file_path)  # 운영체제 권한을 이용해 실제 파일 이름을 즉시 새 이름으로 덮어씁니다.
                    print(f"변경 완료: '{file}' -> '{new_file_name}'")  # 성공적으로 이름이 바뀐 파일을 사용자에게 친절하게 알려줍니다.
                else:  # 이미 새 이름과 완전히 동일한 파일이 존재할 경우 실행되는 예외 처리 블록입니다.
                    print(f"건너뜀 (중복 파일 존재): '{file}' -> '{new_file_name}'")  # 덮어쓰기로 인한 데이터 유실을 막기 위해 작업을 건너뛰었음을 알립니다.

    print("파일명 정규화 작업이 모두 완료되었습니다!\n")  # 함수의 모든 반복 탐색이 끝나고 안전하게 작업이 종료되었음을 알립니다.

# --- 함수 사용 예시 ---
# target_path = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/d03"  # 변경 작업을 수행할 타겟 폴더 경로를 문자열 변수에 저장합니다.
# normalize_filenames(target_path)  # 만들어둔 함수에 경로를 전달하여 파일명 정규화 작업을 곧바로 실행합니다.