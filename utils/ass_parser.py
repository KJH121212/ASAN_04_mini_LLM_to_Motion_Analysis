import os
import re

# 🌟 [추가됨] 시간을 프레임 인덱스로 변환하는 유틸리티 함수
def time_to_frame_index(time_str: str, fps: float) -> int:
    """ "H:MM:SS.cs" 형태의 ASS 시간 문자열을 프레임 번호(정수)로 변환합니다. """
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return int(total_seconds * fps)

def read_ass_subtitles(file_path):
    raw_subtitles = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        for line in lines:
            if line.startswith("Dialogue:"):
                parts = line.split(",", 9)
                start_time = parts[1].strip()
                end_time = parts[2].strip()
                text = parts[9].strip()

                clean_text = re.sub(r"\{.*?\}", "", text)
                clean_text = clean_text.replace("\\N", " ")
                raw_subtitles.append([start_time, end_time, clean_text])
    except FileNotFoundError:
        print(f"❌ [오류] '{file_path}' 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"❌ [오류] 파일 로드 중 예기치 못한 문제가 발생했습니다: {e}")

    return raw_subtitles

# 🌟 [수정됨] 파라미터에 fps를 추가하여, 값이 들어오면 즉시 프레임 번호로 덮어씁니다.
def parse_bayley_subtitle_data(raw_subtitles, fps=None):
    parsed_table_data = []

    for start_time, end_time, text in raw_subtitles:
        action_items = text.split("/")

        for item in action_items:
            item = item.strip()
            if not item: continue

            parts = item.split(",")

            # fps가 제공되었다면 시간(String)을 프레임(Int)으로 변환합니다.
            if fps is not None:
                final_start = time_to_frame_index(start_time, fps)
                final_end = time_to_frame_index(end_time, fps)
            else:
                final_start, final_end = start_time, end_time

            if len(parts) == 3:
                domain = parts[0].strip()
                item_num = parts[1].strip()
                score = parts[2].strip()
                parsed_table_data.append([final_start, final_end, domain, item_num, score])
            else:
                parsed_table_data.append([final_start, final_end, "Unknown", "Unknown", item])

    return parsed_table_data