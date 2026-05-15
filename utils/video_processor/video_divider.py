# video_divider.py
# 영상에서 특정 구간을 자르고 소리와 함께 저장하는 기능을 제공합니다.

from moviepy.editor import VideoFileClip
from pathlib import Path

def cut_video_with_audio(input_path: str, output_path: str, start_time: str, end_time: str):
    """
    영상을 지정된 시간으로 자르고 소리와 함께 저장합니다.
    
    Args:
        input_path (str): 원본 영상 경로
        output_path (str): 저장할 영상 경로
        start_time (str): 시작 시간 ("MM:SS" 또는 "HH:MM:SS" 또는 초 단위 숫자)
        end_time (str): 종료 시간 ("MM:SS" 또는 "HH:MM:SS" 또는 초 단위 숫자)
    """
    try:
        # 1. 영상 로드 (오디오 포함)
        clip = VideoFileClip(input_path)
        
        # 2. 구간 자르기 (subclip은 오디오를 자동으로 동기화하여 함께 자릅니다)
        # start_time과 end_time이 "00:15" 형태라면 moviepy가 자동으로 초로 변환합니다.
        sub_clip = clip.subclip(start_time, end_time)
        
        # 3. 결과 저장
        # codec="libx264"는 범용적인 영상 코덱이며, 
        # audio_codec="aac"를 지정하여 소리 품질과 호환성을 확보합니다.
        sub_clip.write_videofile(
            output_path, 
            codec="libx264", 
            audio_codec="aac",
            temp_audiofile='temp-audio.m4a', 
            remove_temp=True
        )
        
        # 4. 메모리 해제 (매우 중요: 안 하면 나중에 파일 접근 에러가 날 수 있음)
        clip.close()
        sub_clip.close()
        
        print(f"✅ 영상 추출 완료: {output_path} ({start_time} ~ {end_time})")
        
    except Exception as e:
        print(f"❌ 영상 자르기 중 에러 발생: {e}")