# ================================================================== #
# metadata_v1.0.csv를 기준으로 비디오를 split 해서 00_SPLIT_DATA에 저장
# ass가 없을 경우 예외처리
# .sub_check.txt를 통해 이전에 분할을 했는지 체크
# ================================================================== #

import os
import sys
import decord
import subprocess
import pandas as pd
from pathlib import Path

# 1. 프로젝트 최상위 폴더 경로 설정 
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

print(f"📁 프로젝트 루트 경로: {PROJECT_ROOT}")

# 2. 필요한 모듈 및 데이터 로드
from utils.path_list_d03 import path_list_d03
from utils.ass_parser import read_ass_subtitles, parse_bayley_subtitle_data

# 경로 설정
DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data_d03")
SPLIT_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data_d03/00_SPLIT_VIDEO_V2")
CSV_PATH = DATA_DIR / "metadata_v1.0.csv"

if not CSV_PATH.exists():
    print(f"❌ 메타데이터 파일을 찾을 수 없습니다: {CSV_PATH}")
    sys.exit(1)

df = pd.read_csv(str(CSV_PATH))

# common_path에 'gross_motor'가 포함된 경우만 필터링
df = df[df['common_path'].str.contains('gross_motor', case=False, na=False)]
df.reset_index(drop=True, inplace=True)
print(f"🎯 'gross_motor' 대상 영상 수: {len(df)}")


def check_sub_changed(patient_dir, ass_path):
    """
    텍스트 상태 파일을 읽어 자막 파일이 변경되었는지 감지합니다.
    변경되었거나 파일이 없으면 True, 그대로면 False를 반환합니다.
    """
    check_file = patient_dir / ".sub_check.txt"
    if not check_file.exists():
        return True  # 처음 작업하는 폴더인 경우
        
    current_mtime = os.path.getmtime(ass_path)
    current_size = os.path.getsize(ass_path)
    
    try:
        with open(check_file, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
            info = {}
            for line in lines:
                if ":" in line:
                    k, v = line.split(":", 1)
                    info[k.strip()] = v.strip()
                    
        # 기존 기록과 현재 자막 상태 비교
        saved_mtime = float(info.get("modified_time", 0))
        saved_size = int(info.get("filesize", 0))
        
        if current_mtime == saved_mtime and current_size == saved_size:
            return False  # 변경 안 됨 (스킵 가능)
    except:
        pass
        
    return True  # 자막이 수정되었거나 에러가 난 경우 재작업 필요


def save_sub_check_txt(patient_dir, ass_path):
    """
    작업이 완료된 후 폴더 내에 자막 상태 메타데이터 텍스트 파일을 생성합니다.
    """
    check_file = patient_dir / ".sub_check.txt"
    current_mtime = os.path.getmtime(ass_path)
    current_size = os.path.getsize(ass_path)
    
    with open(check_file, "w", encoding="utf-8") as f:
        f.write(f"filepath: {ass_path}\n")
        f.write(f"modified_time: {current_mtime}\n")
        f.write(f"filesize: {current_size}\n")

def split_video_by_ass_smart(df, output_base_dir, fps=30, dry_run=True):
    if dry_run:
        print("\n⚠️ [DRY RUN MODE] 실제 비디오 분할 파일은 생성되지 않으며, 시뮬레이션 결과만 출력됩니다.")
    else:
        print("\n🎬 [REAL EXECUTION] 스마트 체크 기반 비디오 분할 생성을 시작합니다.")
        
    print("=" * 80)
    
    success_count = 0
    skip_count = 0
    no_ass_count = 0
    
    for idx, row in df.iterrows():
        common_path = row['common_path']
        paths = path_list_d03(common_path)
        
        video_file_path = Path(paths['video'])
        ass_file_path = Path(paths['ass'])
        
        # 1. 자막 파일 부재 시 예외 처리
        if not ass_file_path.exists():
            print(f"⏩[{idx}] 자막 파일 없음 ➔ 건너뜀: {ass_file_path.name}")
            no_ass_count += 1
            continue
            
        if not video_file_path.exists():
            print(f"⚠️[{idx}] 원본 영상 파일이 디스크에 없습니다: {video_file_path.name}")
            continue

        patient_output_dir = output_base_dir / common_path
        
        # 2. 텍스트 체크섬 기반 자막 변경 여부 검사
        is_changed = check_sub_changed(patient_output_dir, ass_file_path)
        
        if not is_changed:
            print(f"⏭️ [{idx}] [상태 동일] 자막 변화 없음 ➔ 폴더 통째로 분할 생략: {common_path}")
            skip_count += 1
            continue

        # 3. 자막이 변경되었다면 구버전 파일 싹 청소
        if not dry_run and patient_output_dir.exists():
            print(f"🧹 [{idx}] 자막 변경 감지! 기존 폴더 데이터 초기화 중...")
            for f in patient_output_dir.glob("*"):
                if f.is_file():
                    f.unlink()
        elif dry_run and patient_output_dir.exists():
            print(f"🧹 [{idx}] [예정] 자막 변경 감지 ➔ 기존 폴더 내 모든 파일 청소 예정")

        try:
            # 💡 [해결 책]: 시간 초과 방지를 위해 여기서 decord로 영상 메타데이터(vr)를 명확히 선언합니다.
            vr = decord.VideoReader(str(video_file_path))
            video_total_duration = len(vr) / fps  # 영상의 총 길이(초) 계산
            
            # 자막 읽기 및 전처리
            raw_subtitles = read_ass_subtitles(str(ass_file_path))
            timeline_dict = {}
            
            for sub in raw_subtitles:
                start_t, end_t, text = sub[0], sub[1], sub[2].strip()
                
                lines = text.split("/") if "/" in text else [text]
                for line in lines:
                    line = line.strip()
                    if line.startswith("GM") and not line.startswith("GM,"):
                        line = "GM," + line[2:]
                        
                    parsed = parse_bayley_subtitle_data([(start_t, end_t, line)], fps=fps)
                    if parsed:
                        sf, ef, cat, item_num, score = parsed[0]
                        score = str(score).split(",")[-1].replace("점", "").strip()
                        if score == '?': score = 'unknown'
                        
                        time_key = (sf, ef)
                        if time_key not in timeline_dict:
                            timeline_dict[time_key] = []
                        timeline_dict[time_key].append((cat, item_num, score))

            if not dry_run:
                patient_output_dir.mkdir(parents=True, exist_ok=True)

            print(f"📂 [{idx}] {ass_file_path.name} 분할 커팅 시작... (총 {len(timeline_dict)}개 구간)")

            # FFmpeg 비디오+오디오 동시 컷팅 실행 (마진 포함 안전 연산)
            for (start_frame, end_frame), items in timeline_dict.items():
                raw_start_sec = start_frame / fps
                raw_end_sec = end_frame / fps
                
                # 💡 앞뒤 1초 마진 적용 (0초 미만으로 떨어지거나 총 길이를 넘어가지 않게 가드)
                margin_sec = 1.0
                start_sec = max(0.0, raw_start_sec - margin_sec)
                end_sec = min(video_total_duration, raw_end_sec + margin_sec)
                duration_sec = end_sec - start_sec
                
                name_parts = []
                for cat, item_num, score in items:
                    name_parts.append(f"{cat}_{item_num}_{score}")
                
                output_filename = "__".join(name_parts) + f"__{start_frame}f.mp4"
                output_file_path = patient_output_dir / output_filename
                
                ffmpeg_cmd = [
                    "ffmpeg", "-y", "-ss", str(start_sec), "-t", str(duration_sec),
                    "-i", str(video_file_path), "-c", "copy", str(output_file_path)
                ]
                
                if dry_run:
                    print(f"   ➔ [컷팅 예정] {start_frame:5d} ~ {end_frame:5d} f | {output_filename}")
                else:
                    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
            # 작업이 무사히 끝나면 최신 자막 상태를 텍스트 파일로 저장
            if not dry_run:
                save_sub_check_txt(patient_output_dir, ass_file_path)
                
            success_count += 1
            
        except Exception as e:
            print(f"❌ [{idx}] {ass_file_path.name} 처리 중 에러 발생: {e}")

    print("=" * 80)
    print(f"✨ 완료: {success_count}개 생성/갱신 | {skip_count}개 생략(스마트패스) | {no_ass_count}개 자막 없음 패스")


# --- 스크립트 실행 구간 ---
if __name__ == "__main__":
    # 2. 구조가 완벽하다면 아래 주석을 풀고 실행하여 진짜 데이터셋을 빌드하세요!
    split_video_by_ass_smart(df, SPLIT_DIR, fps=30, dry_run=False)