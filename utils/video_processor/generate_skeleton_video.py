import json
import cv2
import random
import glob
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Union, Optional, Callable
from utils.extract_kpt import normalize_skeleton_array

# ==============================================================================
# ⚙️ 1. 통합 설정 (Configuration)
# ==============================================================================
class Config:
    # 색상 (BGR)
    COLOR_SKELETON = (100, 100, 100)
    COLOR_ID = (0, 255, 0)       # Green
    COLOR_TEXT = (255, 255, 255) # White
    
    # 17 Keypoints (Body) 색상
    COLOR_RIGHT = (0, 0, 255)    # Red
    COLOR_LEFT = (255, 0, 0)     # Blue

    # 133 Keypoints (WholeBody) 색상
    COLOR_BODY = (255, 100, 0)
    COLOR_FACE = (0, 255, 255)
    COLOR_HAND_L = (0, 0, 255)
    COLOR_HAND_R = (255, 0, 0)
    COLOR_FOOT_L = (0, 128, 255)
    COLOR_FOOT_R = (255, 128, 0)

    # 폰트 설정
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    FONT_THICKNESS = 1

    # 17 Keypoints 연결 정보 (COCO)
    LINKS_17 = [
        (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15),
        (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)
    ]
    KPT_17_LEFT = {5, 7, 9, 11, 13, 15}
    KPT_17_RIGHT = {6, 8, 10, 12, 14, 16}

    # 133 Keypoints 연결 정보 (WholeBody)
    LINKS_133 = {
        'body': [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)],
        'feet': [(15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22)],
        'face': [(23, 24), (24, 25), (26, 27), (27, 28), (62, 63), (63, 64), (64, 65), (65, 66)],
        'l_hand': [(9, 91), (91, 92), (92, 93), (93, 94), (94, 95), (91, 96), (96, 97), (97, 98), (98, 99), (91, 100), (100, 101), (101, 102), (102, 103), (91, 104), (104, 105), (105, 106), (106, 107), (91, 108), (108, 109), (109, 110), (110, 111)],
        'r_hand': [(10, 112), (112, 113), (113, 114), (114, 115), (115, 116), (112, 117), (117, 118), (118, 119), (119, 120), (112, 121), (121, 122), (122, 123), (123, 124), (112, 125), (125, 126), (126, 127), (127, 128), (112, 129), (129, 130), (130, 131), (131, 132)]  
    }

    LINKS_12 = [
        (0, 2), (2, 4), # 왼쪽 팔 (Shoulder-Elbow-Wrist)
        (1, 3), (3, 5), # 오른쪽 팔
        (6, 8), (8, 10),# 왼쪽 다리 (Hip-Knee-Ankle)
        (7, 9), (9, 11),# 오른쪽 다리
        (0, 1),         # 어깨 가로선
        (6, 7),         # 골반 가로선
        (0, 6), (1, 7)  # 몸통 세로선 (어깨-골반)
    ]

    # 왼쪽/오른쪽 구분을 위한 인덱스 집합 (12포인트 기준)
    KPT_12_LEFT = {0, 2, 4, 6, 8, 10}  # 짝수 (기존 5, 7, 9, 11, 13, 15)
    KPT_12_RIGHT = {1, 3, 5, 7, 9, 11} # 홀수 (기존 6, 8, 10, 12, 14, 16)

# ==============================================================================
# 🎨 2. 그리기 유틸리티 (Visualization Utils)
# ==============================================================================
class Visualizer:
    @staticmethod
    def draw_text_with_bg(img: np.ndarray, text: str, pos: tuple, 
                          bg_color=(0, 0, 0), txt_color=(255, 255, 255), align_bottom=False):
        """텍스트 뒤에 배경 박스를 그려 가독성을 높입니다. (화면 밖으로 나가는 문제 해결)"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 1
        (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
        x, y = int(pos[0]), int(pos[1]) # 정수형 변환 보장
        
        img_h, img_w = img.shape[:2]

        # 1. 텍스트 박스의 크기와 초기 시작 좌표(x1, y1) 계산
        if align_bottom:
            box_w, box_h = w + 20, h + 15
            x1, y1 = x, y - h - 10
        else:
            box_w, box_h = w + 10, h + 10
            x1, y1 = x, y - h - 5

        # 2. 화면 경계(Boundary)를 벗어나지 않도록 좌표 보정 (Clamping)
        x1 = max(0, min(x1, img_w - box_w))
        y1 = max(0, min(y1, img_h - box_h))
        x2, y2 = x1 + box_w, y1 + box_h

        # 3. 보정된 박스 위치에 따라 텍스트의 베이스라인(Base-line) 재조정
        if align_bottom:
            text_x, text_y = x1 + 10, y2 - 5
        else:
            text_x, text_y = x1 + 5, y2 - 5

        # 4. 최종 그리기
        cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
        cv2.putText(img, text, (text_x, text_y), font, scale, txt_color, thickness, cv2.LINE_AA)

    @staticmethod
    def draw_bbox_and_id(img: np.ndarray, bbox: list, obj_id: Union[int, str], color: tuple):
        """BBox와 ID를 그립니다."""
        if bbox is None: return

        # 평탄화 (Flatten)
        bbox_flat = np.array(bbox).flatten()
        if len(bbox_flat) < 4: return
        
        x1, y1, x2, y2 = map(int, bbox_flat[:4])
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        Visualizer.draw_text_with_bg(img, f"ID: {obj_id}", (xmin, ymin), bg_color=color)

    @staticmethod
    def rle_to_mask(rle: List[int], height: int, width: int) -> np.ndarray:
        mask = np.zeros(height * width, dtype=np.uint8)
        if not rle: return mask.reshape((height, width))
        rle = np.array(rle)
        starts = rle[0::2] - 1
        lengths = rle[1::2]
        ends = starts + lengths
        for lo, hi in zip(starts, ends):
            mask[max(lo, 0):min(hi, len(mask))] = 1
        return mask.reshape((height, width))
    
    @staticmethod
    def apply_face_mosaic(img: np.ndarray, keypoints: np.ndarray, conf_threshold: float = 0.0):
        """17 또는 133 키포인트에서 얼굴 부위(0~4)를 추출하여 모자이크(가우시안 블러)를 적용합니다."""
        if len(keypoints) < 17: 
            return img
        
        # 0:코, 1:왼눈, 2:오른눈, 3:왼귀, 4:오른귀
        face_kpts = keypoints[:5] 
        coords = face_kpts[:, :2].astype(int)
        scores = face_kpts[:, 2] if face_kpts.shape[1] > 2 else np.ones(5)
        
        # 신뢰도가 높은 유효한 좌표만 필터링
        valid_mask = (scores > conf_threshold) & (coords[:, 0] > 0) & (coords[:, 1] > 0)
        valid_coords = coords[valid_mask]
        
        if len(valid_coords) == 0: 
            return img
        
        # 얼굴 영역 BBox 계산
        xmin, ymin = np.min(valid_coords, axis=0)
        xmax, ymax = np.max(valid_coords, axis=0)
        
        # 패딩 추가
        pad_x = 30
        pad_y_top = 40    
        pad_y_bottom = 30 
        
        img_h, img_w = img.shape[:2]
        xmin = max(0, xmin - pad_x)
        ymin = max(0, ymin - pad_y_top)
        xmax = min(img_w, xmax + pad_x)
        ymax = min(img_h, ymax + pad_y_bottom)
        
        # 가우시안 블러 모자이크 적용
        if xmax > xmin and ymax > ymin:
            roi = img[ymin:ymax, xmin:xmax]
            blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30) 
            img[ymin:ymax, xmin:xmax] = blurred_roi
            
        return img
    
# ==============================================================================
# 🚀 3. 핵심 엔진 (Video Engine)
# ==============================================================================
def create_video_engine(
    frame_dir: Union[str, Path],
    output_path: Union[str, Path],
    json_dir: Union[str, Path],
    draw_callback: Callable[[np.ndarray, dict], np.ndarray],
    fps: int = 30,
    start_idx: int = 0,
    end_idx: Optional[int] = None
):
    frame_path = Path(frame_dir)
    save_path = Path(output_path)
    json_path = Path(json_dir)

    all_frames = sorted(list(frame_path.rglob("*.jpg")) + list(frame_path.rglob("*.png")), 
                        key=lambda x: int(''.join(filter(str.isdigit, x.stem)))) 
    
    json_map = {f.stem: f for f in json_path.rglob("*.json")}

    end_idx = min(len(all_frames), end_idx) if end_idx is not None else len(all_frames)
    start_idx = max(0, start_idx)
    target_frames = all_frames[start_idx:end_idx]
    
    if not target_frames:
        print("❌ [Error] 처리할 프레임 구간이 유효하지 않습니다.")
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)
    first_img = cv2.imread(str(target_frames[0]))
    h, w = first_img.shape[:2]
    out = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    print(f"🎬 비디오 생성: {save_path.name} (Range: {start_idx}~{end_idx}, Frames: {len(target_frames)})")

    for i, frame_file in enumerate(tqdm(target_frames, desc="Processing")):
        frame = cv2.imread(str(frame_file))
        if frame is None: continue

        json_file = json_map.get(frame_file.stem)
        data = {}
        if json_file and json_file.exists():
            try:
                with open(json_file, 'r') as f: data = json.load(f)
            except: pass

        frame = draw_callback(frame, data)

        real_idx = start_idx + i
        Visualizer.draw_text_with_bg(
            frame, 
            f"Frame: {real_idx}/{len(all_frames)} (Seg: {i}/{len(target_frames)})", 
            (20, h - 20), 
            align_bottom=True
        )

        out.write(frame)

    out.release()
    print(f"✅ 완료: {save_path}")

def generate_integrated_video(
    frame_dir: Union[str, Path],
    output_path: Union[str, Path],
    skeleton_dir: Optional[Union[str, Path]] = None,
    sam_dir: Optional[Union[str, Path]] = None,
    target_ids: Optional[List[Union[int, str]]] = None,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    conf_threshold: float = 0.0,
    fps: int = 30,
    apply_mosaic: bool = False,
    draw_skeleton: bool = True
):
    """
    프레임 이미지에 스켈레톤(17kpt)과 SAM 마스크를 통합하여 오버레이 비디오를 생성합니다.
    """
    _sam_color_cache = {} 
    
    def get_sam_color(oid):
        if oid not in _sam_color_cache:
            _sam_color_cache[oid] = [random.randint(50, 255) for _ in range(3)]
        return _sam_color_cache[oid]

    def integrated_drawer(frame, data_dict):
        h_img, w_img = frame.shape[:2]

        # 1. SAM 데이터 처리
        sam_data = data_dict.get('sam', {})
        if "objects" in sam_data:
            overlay = frame.copy()
            for obj in sam_data["objects"]:
                obj_id = obj.get("id", "?")
                if target_ids and obj_id not in target_ids: continue
                
                color = get_sam_color(obj_id)
                rle = obj.get("segmentation", {}).get("counts")
                
                if rle:
                    mask = Visualizer.rle_to_mask(rle, h_img, w_img)
                    if mask.sum() > 0:
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(overlay, contours, -1, color, 2)
                        cv2.fillPoly(overlay, contours, color)
                        
                        ys, xs = np.where(mask)
                        xmin, ymin, xmax, ymax = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)
                        Visualizer.draw_text_with_bg(frame, f"SAM ID: {obj_id}", (xmin, ymax + 20), bg_color=(255, 255, 255), txt_color=(0, 0, 0))
            
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # 2. Skeleton 데이터 처리
        skel_data = data_dict.get('skeleton', {})
        for inst in skel_data.get('instance_info', []):
            obj_id = inst.get('instance_id', inst.get('id', '?'))
            if target_ids and obj_id not in target_ids: continue
            if inst.get('score', 1.0) < conf_threshold: continue

            kpts = np.array(inst.get('keypoints', []))
            if kpts.shape[0] == 0: continue

            # ⭐️ 모자이크 옵션이 True일 경우 적용
            if apply_mosaic:
                frame = Visualizer.apply_face_mosaic(frame, kpts, conf_threshold)

            # ⭐️ 옵션이 True일 때만 좌표를 계산하고 뼈대와 BBox를 그립니다.
            if draw_skeleton:
                coords = kpts[:, :2].astype(int)
                scores = inst.get('keypoint_scores', kpts[:, 2] if kpts.shape[1] > 2 else np.ones(len(coords)))

                for u, v in Config.LINKS_17:
                    if u < len(coords) and v < len(coords) and scores[u] > conf_threshold and scores[v] > conf_threshold:
                        cv2.line(frame, tuple(coords[u]), tuple(coords[v]), Config.COLOR_SKELETON, 2, cv2.LINE_AA)
                for idx, (x, y) in enumerate(coords):
                    if scores[idx] > conf_threshold and x > 0:
                        dot_color = Config.COLOR_RIGHT if idx in Config.KPT_17_RIGHT else (Config.COLOR_LEFT if idx in Config.KPT_17_LEFT else (0, 255, 0))
                        cv2.circle(frame, (x, y), 4, dot_color, -1, cv2.LINE_AA)

                bbox = inst.get('bbox')
                if bbox:
                    bbox_flat = np.array(bbox).flatten().astype(int)
                    xmin, ymin, xmax, ymax = bbox_flat[:4]
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 0), 2)
                    Visualizer.draw_text_with_bg(frame, f"SKEL ID: {obj_id}", (xmin, ymin - 10), bg_color=(0, 0, 0), txt_color=(255, 255, 255))
        return frame

    frame_path = Path(frame_dir)
    frame_files = sorted(list(frame_path.rglob("*.jpg")) + list(frame_path.rglob("*.png")),
                         key=lambda x: int(''.join(filter(str.isdigit, x.stem))))
    
    final_end = end_idx if end_idx is not None else len(frame_files)
    target_frames = frame_files[start_idx:final_end]
    
    if not target_frames: 
        print("❌ [Error] 렌더링할 프레임이 없습니다.")
        return

    skel_map = {f.stem: f for f in Path(skeleton_dir).rglob("*.json")} if skeleton_dir else {}
    sam_map = {f.stem: f for f in Path(sam_dir).rglob("*.json")} if sam_dir else {}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    h, w = cv2.imread(str(target_frames[0])).shape[:2]
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for i, f_file in enumerate(tqdm(target_frames, desc="Integrated Rendering")):
        frame = cv2.imread(str(f_file))
        combined_data = {'skeleton': {}, 'sam': {}}

        skel_json_path = skel_map.get(f_file.stem)
        if skel_json_path and skel_json_path.exists(): 
            try:
                with open(skel_json_path, 'r') as f:
                    combined_data['skeleton'] = json.load(f)
            except: pass

        sam_json_path = sam_map.get(f_file.stem)
        if sam_json_path and sam_json_path.exists():
            try:
                with open(sam_json_path, 'r') as f:
                    combined_data['sam'] = json.load(f)
            except: pass

        frame = integrated_drawer(frame, combined_data)
        
        info = f"Frame: {start_idx + i} / {len(frame_files)} | ID Filter: {target_ids if target_ids else 'ALL'}"
        Visualizer.draw_text_with_bg(frame, info, (20, h - 20), align_bottom=True)

        out.write(frame)

    out.release()
    print(f"✅ 통합 비디오 생성 완료: {output_path}")


def generate_skeleton_video_np(
    frame_dir: Union[str, Path],
    output_path: Union[str, Path],
    skeleton_np: Optional[np.ndarray] = None, 
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    conf_threshold: float = 0.0,
    fps: int = 30,
    apply_mosaic: bool = False,
    draw_skeleton: bool = True
):
    """
    Numpy 스켈레톤 데이터를 프레임 이미지 위에 그려 비디오를 생성합니다.
    """
    def integrated_drawer(frame, skel_frame_data): 
        h_img, w_img = frame.shape[:2] 

        if skel_frame_data is not None and np.any(skel_frame_data):
            num_kpts = skel_frame_data.shape[0]
            # ⭐️ 모자이크 적용
            if apply_mosaic:
                frame = Visualizer.apply_face_mosaic(frame, skel_frame_data, conf_threshold)

            # ⭐️ 스켈레톤 및 박스 그리기
            if draw_skeleton:
                coords = skel_frame_data[:, :2].astype(int)
                scores = skel_frame_data[:, 2] if skel_frame_data.shape[1] > 2 else np.ones(num_kpts)

                if num_kpts == 12:
                    links = Config.LINKS_12
                    left_idx, right_idx = Config.KPT_12_LEFT, Config.KPT_12_RIGHT
                else:
                    links = Config.LINKS_17
                    left_idx, right_idx = Config.KPT_17_LEFT, Config.KPT_17_RIGHT

                for u, v in links:
                    if u < num_kpts and v < num_kpts:
                        if scores[u] > conf_threshold and scores[v] > conf_threshold:
                            if coords[u][0] > 0 and coords[v][0] > 0:
                                cv2.line(frame, tuple(coords[u]), tuple(coords[v]), Config.COLOR_SKELETON, 2, cv2.LINE_AA)

                for idx, (x, y) in enumerate(coords):
                    if scores[idx] > conf_threshold and x > 0:
                        dot_color = Config.COLOR_RIGHT if idx in right_idx else (Config.COLOR_LEFT if idx in left_idx else (0, 255, 0))
                        cv2.circle(frame, (x, y), 4, dot_color, -1, cv2.LINE_AA)

                valid_mask = (scores > conf_threshold) & (coords[:, 0] > 0)
                if np.any(valid_mask):
                    v_coords = coords[valid_mask]
                    xmin, ymin = np.min(v_coords, axis=0)
                    xmax, ymax = np.max(v_coords, axis=0)
                    cv2.rectangle(frame, (int(xmin-15), int(ymin-15)), (int(xmax+15), int(ymax+15)), (0, 0, 0), 2)
                    Visualizer.draw_text_with_bg(frame, "SKEL NP", (int(xmin-15), int(ymin-25)), bg_color=(0, 0, 0), txt_color=(255, 255, 255))
        return frame

    frame_path = Path(frame_dir)
    frame_files = sorted(list(frame_path.rglob("*.jpg")) + list(frame_path.rglob("*.png")),
                         key=lambda x: int(''.join(filter(str.isdigit, x.stem))))
    
    final_end = end_idx if end_idx is not None else len(frame_files)
    target_frames = frame_files[start_idx:final_end]
    
    if not target_frames: return

    num_to_process = len(target_frames)
    if skeleton_np is not None:
        num_to_process = min(num_to_process, len(skeleton_np))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    first_img = cv2.imread(str(target_frames[0]))
    h, w = first_img.shape[:2]
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for i in tqdm(range(num_to_process), desc="Rendering Skeleton Video"):
        frame = cv2.imread(str(target_frames[i]))
        if frame is None: continue

        current_skel = skeleton_np[i] if skeleton_np is not None else None
        frame = integrated_drawer(frame, current_skel)
        
        info_text = f"Frame: {start_idx + i} | Kpts: {current_skel.shape[0] if current_skel is not None else 0}"
        Visualizer.draw_text_with_bg(frame, info_text, (20, h - 20), align_bottom=True)

        out.write(frame)

    out.release()
    print(f"✅ 비디오 저장 완료: {output_path}")