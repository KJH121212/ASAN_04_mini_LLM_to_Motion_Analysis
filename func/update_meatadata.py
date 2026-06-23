# ================================================================== #
# metadata_v1.0.csv 이후 metadata_v1.1.csv 로 업데이트
# 파일명 수정 (띄어쓰기->_, 대문자->소문자)
# meatadata 업데이트 및 이름순 정렬
# ================================================================== #

import os
import pandas as pd
from pathlib import Path

def run_complete_integrated_update(base_path: str, ass_base_path: str, csv_input_path: str, csv_output_path: str, dry_run: bool = True):
    
    root = Path(base_path)
    if not root.exists():
        print(f"❌ 원본 경로가 존재하지 않습니다: {base_path}")
        return

    print(f"==================================================")
    print(f"🎬 1단계: 디스크 파일 및 폴더 이름 변경 시작 (dry_run={dry_run})")
    print(f"==================================================")
    
    # 1. 디스크 파일/폴더 변경 (깊은 하위 경로 위주로 역순 정렬하여 안전하게 이름 변경)
    all_paths = sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True)
    rename_count = 0
    
    for path in all_paths:
        old_name = path.name
        # 규칙: 띄어쓰기는 _로, 대문자는 소문자로
        new_name = old_name.replace(" ", "_").lower()
        
        if old_name == new_name:
            continue
            
        new_path = path.with_name(new_name)
        type_str = "폴더" if path.is_dir() else "파일"
        
        if not dry_run:
            try:
                path.rename(new_path)
            except Exception as e:
                print(f"   ❌ [변경 실패] {path.name} -> {new_name} ({e})")
                continue
        else:
            print(f"   [예정] {type_str}: {path.relative_to(root)} ➔ {new_name}")
            
        rename_count += 1
        
    print(f"➔ 총 {rename_count}개의 파일/폴더 이름 변경 완료(또는 예정).")

    # dry_run 일 때는 실제로 파일명이 안 바뀌었으므로 2단계를 진행하면 매칭이 안 됩니다.
    # 구조만 확인하실 수 있게 안내하고 계속 진행합니다.
    if dry_run:
        print("\n⚠️ [안내] 현재 Dry Run 모드입니다. 2단계는 '실제 이름이 변경되었다고 가정'하고 가상으로 파싱합니다.")

    print(f"\n==================================================")
    print(f"🔍 2단계: BASE_PATH 내의 모든 MP4 파일 새로 수집")
    print(f"==================================================")
    
    # 실제 디스크에 있는 (혹은 변경될 예정인) 모든 mp4 파일 수집
    current_videos = []
    
    # os.walk 대신 pathlib를 사용해 대소문자 변환 규칙이 적용된 기준(또는 적용 전 기준)으로 수집
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".mp4":
            # 실제 rename 규칙이 적용된 후의 경로를 가상/실제로 계산
            rel_parts = [part.replace(" ", "_").lower() for part in p.relative_to(root).parts]
            v_common_path = "/".join(rel_parts)[:-4] # .mp4 제외
            v_video_path = os.path.join(base_path, "/".join(rel_parts))
            v_ass_path = os.path.join(ass_base_path, f"{v_common_path}.ass")
            
            current_videos.append({
                "video_path": v_video_path,
                "common_path": v_common_path,
                "ass_path": v_ass_path
            })
            
    df_current = pd.DataFrame(current_videos)
    print(f"➔ 디스크에서 총 {len(df_current)}개의 동영상 파일을 발견했습니다.")

    print(f"\n==================================================")
    print(f"📊 3단계: 기존 메타데이터와 결합 및 이름순 정렬")
    print(f"==================================================")
    
    if os.path.exists(csv_input_path):
        # 기존 메타데이터 읽기
        df_old = pd.read_csv(csv_input_path)
        
        # 기존 메타데이터의 common_path도 소문자/언더바 규칙 적용
        def sanitize_cp(cp):
            if pd.isna(cp): return cp
            return "/".join([part.replace(" ", "_").lower() for part in str(cp).split('/')])
        df_old['common_path'] = df_old['common_path'].apply(sanitize_cp)
        
        # 경로 관련 컬럼들을 제외한 기존 데이터 컬럼들만 추출 (n_frames, frames_done 등 상태값 보존 목적)
        # video_path, ass_path 컬럼은 새 디스크 기준으로 덮어쓸 것이므로 드롭
        cols_to_drop = [c for c in ['video_path', 'ass_path'] if c in df_old.columns]
        df_old_clean = df_old.drop(columns=cols_to_drop)
        
        # 현재 디스크 전체 목록(df_current)을 기준으로 기존 데이터(df_old_clean)를 Left Join 매칭
        df_final = pd.merge(df_current, df_old_clean, on="common_path", how="left")
    else:
        print(f"⚠️ 기존 메타데이터({csv_input_path})가 없어 전체 새로 생성합니다.")
        df_final = df_current

    # 새롭게 추가된 동영상들의 빈 칸(NaN)을 기본값(0 또는 False)으로 채워주기
    fill_values = {
        "n_frames": 0, "n_json": 0,
        "frames_done": False, "sapiens_done": False, "reextract_done": False, "overlay_done": False,
        "is_train": False, "is_val": False, "id_done": False, "sam_done": False, "split_video": False
    }
    for col, val in fill_values.items():
        if col in df_final.columns:
            if df_final[col].dtype == 'bool':
                df_final[col] = df_final[col].fillna(False).astype(bool)
            else:
                df_final[col] = df_final[col].fillna(val)

    # 지정하신 컬럼 순서 이쁘게 재배치 (컬럼이 존재할 때만 순서 고정)
    desired_order = ["video_path", "common_path", "ass_path", "n_frames", "n_json", 
                     "frames_done", "sapiens_done", "reextract_done", "overlay_done", 
                     "is_train", "is_val", "id_done", "sam_done", "split_video"]
    actual_order = [c for c in desired_order if c in df_final.columns] + [c for c in df_final.columns if c not in desired_order]
    df_final = df_final[actual_order]

    # ✨ 요구사항: video_path 기준 이름순 정렬 및 인덱스 초기화
    df_final.sort_values(by="video_path", ascending=True, inplace=True)
    df_final.reset_index(drop=True, inplace=True)
    
    # 4. 결과 저장
    if not dry_run:
        df_final.to_csv(csv_output_path, index=False)
        print(f"✨ 업데이트 완료! 전체 {len(df_final)}개의 동영상이 반영된 새 메타데이터가 '{csv_output_path}'에 저장되었습니다.")
    else:
        print(f"⚠️ [DRY RUN] 실제 CSV 파일은 저장되지 않았습니다.")
        
    print("\n--- 결과 데이터 예시 (상위 5개) ---")
    print(df_final.head(5))


# --- 실행 환경 설정 ---
if __name__ == "__main__":
    BASE_PATH = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/d03"
    ASS_BASE_PATH = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/d03/ass"
    INPUT_CSV = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data_d03/metadata_v1.0.csv"          # 기존 원본 메타데이터
    # OUTPUT_CSV = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data_d03/metadata_v1.1.csv"  # 최종 갱신되어 저장될 경로
    OUTPUT_CSV= "./data.csv"
    
    run_complete_integrated_update(BASE_PATH, ASS_BASE_PATH, INPUT_CSV, OUTPUT_CSV, dry_run=False)