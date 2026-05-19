# video_divider.py
# 영상에서 특정 구간을 자르고, 필요 시 JSON 관절 데이터를 맵핑해 얼굴 모자이크를 입힌 뒤 소리와 함께 저장합니다.

import cv2
import json
from moviepy.editor import VideoFileClip
from pathlib import Path

def time_to_frame_index(time_val, fps: float) -> int:
    """ 시간 문자열("HH:MM:SS.f" 또는 초 단위)을 프레임 번호로 변환합니다. """
    if isinstance(time_val, str) and ':' in time_val:
        parts = time_val.split(':')
        total_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    else:
        total_seconds = float(time_val)
    return int(total_seconds * fps)

def cut_mosaic_video_with_audio(input_path: str, output_path: str, start_time: str, end_time: str, json_dir_path: str = None, pad_x: int = 20, pad_y: int = 70, blur_strength: int = 45):
    """
    영상을 지정된 시간으로 자르고 소리와 함께 저장합니다. JSON 폴더가 제공되면 실시간 모자이크를 씌웁니다.
    """
    try:
        # 1. 영상 로드 (오디오 포함)
        clip = VideoFileClip(input_path)
        fps = clip.fps
        
        # 2. 구간 자르기 (subclip은 오디오를 자동으로 동기화하여 함께 자릅니다)
        sub_clip = clip.subclip(start_time, end_time)
        
        # 3. 모자이크 옵션이 켜져 있을 경우 (JSON 폴더 경로가 제공된 경우)
        if json_dir_path:
            start_frame_offset = time_to_frame_index(start_time, fps)  # 원본 영상 기준 시작 프레임 번호 역산
            json_base_dir = Path(json_dir_path)
            
            if blur_strength % 2 == 0: blur_strength += 1  # 블러 커널 홀수 보정
            
            # MoviePy 렌더링 파이프라인에 주입할 실시간 프레임 가공 함수
            def process_frame(get_frame, t):
                # get_frame(t)는 시간 t에서의 프레임을 RGB 넘파이 배열로 반환합니다. 복사본을 만들어 원본 훼손을 막습니다.
                frame = get_frame(t).copy()  
                
                # t(초)를 기반으로 원본 영상의 절대 프레임 번호를 추적합니다.
                current_frame_idx = start_frame_offset + int(t * fps)
                
                # [핵심 수정]: 10,000개 단위의 하위 폴더 구조를 수학적으로 계산하여 경로를 찾습니다.
                # 예: 15432 프레임 -> 15432 // 10000 = 1 -> '01' 폴더
                subfolder_idx = current_frame_idx // 10000  
                subfolder_name = f"{subfolder_idx:02d}"  # '00', '01' 형태로 두 자리 패딩을 맞춥니다.
                json_file_name = f"{current_frame_idx:06d}.json"  # '015432.json' 등 6자리 파일명 생성
                
                # 베이스 경로 + 하위 폴더('00', '01') + 파일명(.json) 결합
                target_json_path = json_base_dir / subfolder_name / json_file_name
                
                if target_json_path.exists():
                    with open(target_json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                        
                    if json_data.get("instance_info") and json_data["instance_info"][0].get("keypoints"):
                        face_keypoints = json_data["instance_info"][0]["keypoints"][0:5]
                        
                        x_coords, y_coords = [kp[0] for kp in face_keypoints], [kp[1] for kp in face_keypoints]
                        min_x, max_x = min(x_coords), max(x_coords)
                        min_y, max_y = min(y_coords), max(y_coords)
                        
                        # 인자로 받은 pad_x, pad_y를 활용하여 직사각형 형태의 안전 영역 구축
                        startX, startY = int(min_x - pad_x), int(min_y - pad_y)
                        endX, endY = int(max_x + pad_x), int(max_y + pad_y)
                        
                        frame_height, frame_width = frame.shape[:2]
                        
                        # 픽셀 좌표가 영상 해상도 밖으로 나가지 않도록 한계선을 클리핑합니다.
                        safe_startX, safe_startY = max(0, startX), max(0, startY)
                        safe_endX, safe_endY = min(frame_width - 1, endX), min(frame_height - 1, endY)
                        
                        if safe_endX > safe_startX and safe_endY > safe_startY:
                            face_roi = frame[safe_startY:safe_endY, safe_startX:safe_endX]
                            blurred_face = cv2.GaussianBlur(face_roi, (blur_strength, blur_strength), 0)
                            frame[safe_startY:safe_endY, safe_startX:safe_endX] = blurred_face
                            
                return frame
            
            # fl 메서드를 통해 자르기 스트림에 모자이크 필터를 결합합니다.
            final_clip = sub_clip.fl(process_frame)
        else:
            # JSON 폴더가 제공되지 않으면 원본 그대로 둡니다.
            final_clip = sub_clip
        
        # 4. 결과 인코딩 및 디스크 저장 (단 한 번만 수행됨)
        final_clip.write_videofile(
            output_path, 
            codec="libx264", 
            audio_codec="aac",
            temp_audiofile='temp-audio.m4a', 
            remove_temp=True,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-bf", "0"]  # 짧은 클립 에러(NAL unit) 방지 옵션 추가
        )
        
        # 5. 메모리 해제 (I/O 병목 방지)
        final_clip.close()
        if json_dir_path:
            sub_clip.close()
        clip.close()
        
        print(f"✅ 단일 패스 처리 완료(모자이크: {bool(json_dir_path)}): {output_path}")
        
    except Exception as e:
        print(f"❌ 영상 전처리 중 에러 발생: {e}")