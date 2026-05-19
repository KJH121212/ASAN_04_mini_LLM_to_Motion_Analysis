# from utils.ass_parser import read_ass_subtitles,parse_bayley_subtitle_data


import os  # 임시 테스트용 .ass 자막 파일을 생성하고 삭제하기 위해 가져옵니다.
import re  # 정규표현식을 사용해 자막 내 특수 태그 제거 및 데이터 분리를 위해 가져옵니다.

def read_ass_subtitles(file_path):  # 파일 경로를 받아 기본적인 시간과 텍스트를 추출하는 함수입니다.
    raw_subtitles = []  # [[시작시간, 종료시간, 원본텍스트], ...] 구조로 담을 빈 리스트입니다.

    try:  # 파일 미존재 및 파일 읽기 에러에 대응하기 위해 예외 블록을 시작합니다.
        with open(
            file_path, "r", encoding="utf-8"
        ) as file:  # 파일을 인코딩 컨텍스트 매니저로 안전하게 열어줍니다.
            lines = (
                file.readlines()
            )  # 파일의 모든 텍스트 줄을 한 번에 읽어와 배열로 보관합니다.

        for (
            line
        ) in (
            lines
        ):  # 자막 파일의 라인을 한 줄씩 검사하여 대사 행만 골라냅니다.
            if line.startswith(
                "Dialogue:"
            ):  # 실제 화면에 찍히는 대사(Dialogue) 이벤트 라인인지 판별합니다.
                parts = line.split(
                    ",", 9
                )  # 대사 텍스트 내의 유실을 막기 위해 콤마 기준 최대 9번만 분리합니다.

                start_time = parts[
                    1
                ].strip()  # 2번째 세그먼트에서 앞뒤 공백을 자르고 자막 시작 시간을 보관합니다.
                end_time = parts[
                    2
                ].strip()  # 3번째 세그먼트에서 앞뒤 공백을 자르고 자막 종료 시간을 보관합니다.
                text = parts[
                    9
                ].strip()  # 10번째 세그먼트에서 특수 효과가 섞인 오리지널 대사 본문을 가져옵니다.

                # 자막 제어용 특수 태그(예: {\pos(192,540)})를 정규식으로 청소합니다.
                clean_text = re.sub(
                    r"\{.*?\}", "", text
                )  # 중괄호 패턴과 내부 매칭 문자열을 공백으로 소거합니다.
                clean_text = clean_text.replace(
                    "\\N", " "
                )  # .ass 파일용 강제 개행 문자를 가독성을 위해 한 칸 공백으로 치환합니다.

                raw_subtitles.append(
                    [start_time, end_time, clean_text]
                )  # 가공하기 좋게 다듬어진 행 셋을 반환용 리스트에 보관합니다.

    except FileNotFoundError:  # 명시된 위치에 자막 파일이 유실되어 데이터 접근이 불가능할 때 잡힙니다.
        print(
            f"❌ [오류] '{file_path}' 파일을 찾을 수 없습니다."
        )  # 예외 종료 전 사용자에게 안내 메시지를 띄웁니다.
    except Exception as e:  # 원인 미상의 포맷 붕괴나 스트림 장애를 포착합니다.
        print(
            f"❌ [오류] 파일 로드 중 예기치 못한 문제가 발생했습니다: {e}"
        )  # 시스템 장애 로그를 출력합니다.

    return raw_subtitles  # 정제 가공 처리가 끝난 2차원 리스트 결과물을 최종 리턴합니다.


# ==========================================
# [함수 2] 함수 1의 결과값을 입력받아 내용을 정밀 파싱하고 표로 출력하는 함수
# ==========================================
def parse_bayley_subtitle_data(raw_subtitles):  # 넘겨받은 유효 자막 데이터를 도메인별로 쪼갭니다.
    parsed_table_data = []  # [[시작, 종료, 도메인, 문항, 점수], ...] 구조로 담을 결과 리스트입니다.

    for (
        start_time,
        end_time,
        text,
    ) in (
        raw_subtitles
    ):  # 전달받은 리스트 내부의 요소들을 루프를 돌며 디스트럭처링합니다.
        # 슬래시(/)를 기준으로 다중 관찰된 베일리 검사 항목들을 1차로 쪼갭니다.
        action_items = text.split(
            "/"
        )  # '/' 문자를 스플릿하여 리스트 배열 형태로 가공합니다.

        for (
            item
        ) in (
            action_items
        ):  # 슬래시로 분리된 개별 액션(예: 'GM,36,2') 단위를 순회합니다.
            item = item.strip()
            if (
                not item
            ):  # 혹시 모를 자막 오기입 공백(// 등)이 있다면 제외합니다.
                continue

            # 쉼표(,)를 기준으로 대분류, 하위동작 번호, 점수를 2차로 분리합니다.
            parts = item.split(",")

            if (
                len(parts) == 3
            ):  # 정상 규격 포맷인 [도메인, 문항 ID, 스코어] 구조를 충족하는지 검증합니다.
                domain = parts[0].strip()  # 대분류 동작 유형 코드를 추출합니다.
                item_num = parts[1].strip()  # 세부 하위 동작 번호를 추출합니다.
                score = parts[2].strip()  # 해당 문항 결과 점수 값을 추출합니다.

                # 수집된 시점 정보와 쪼개진 동작 속성들을 하나의 행 데이터로 구조화하여 추가합니다.
                parsed_table_data.append(
                    [start_time, end_time, domain, item_num, score]
                )
            else:  # 형식이 깨진 예외적인 상황인 경우 처리합니다.
                parsed_table_data.append(
                    [start_time, end_time, "Unknown", "Unknown", item]
                )

    return parsed_table_data  # 완전히 구조화된 데이터프레임 형태의 리스트를 반환합니다.