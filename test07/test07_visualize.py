# ==============================================================================
# 📊 [실험 B-1 시각화 최종] 동작 번호별 개수(n) 카운트 및 다중 플로팅 적용
# ==============================================================================

import re
import numpy as np
import umap
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import defaultdict

# ==============================================================================
# 📂 1. 데이터 로드 및 차원 축소
# ==============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
LOAD_PATH = PROJECT_ROOT / "test07" / "bayley_features.npz"
SAVE_DIR = PROJECT_ROOT / "test07"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

data = np.load(LOAD_PATH)
features = data['embeddings'] # [N, 512]
raw_labels = data['labels']   # [N]

print(f"✅ 데이터 로드 완료! 총 {features.shape[0]}개의 클립 벡터")
from sklearn.preprocessing import normalize # 상단 import 부분에 추가해도 되고 여기에 둬도 됩니다.

print("🎨 차원 축소 연산을 시작합니다 (정규화 및 노이즈 추가)...")

# 💡 1. L2 정규화: 트랜스포머의 거대한 피처 스케일을 1로 통일
features_normalized = normalize(features, norm='l2')

# 💡 2. 미세 노이즈 추가: 완벽히 똑같은 "검은 화면" 벡터들 때문에 
# t-SNE와 PCA가 고장나는 것을 막기 위해 초미세 노이즈를 더해줍니다.
np.random.seed(42)
noise = np.random.normal(0, 1e-5, features_normalized.shape)
features_ready = features_normalized + noise

# 💡 3. 원본 features 대신 features_ready를 사용하여 차원 축소 진행
embeddings_2d = {
    "UMAP": umap.UMAP(n_neighbors=15, min_dist=0.3, metric='cosine', random_state=42).fit_transform(features_ready),
    "t-SNE": TSNE(n_components=2, perplexity=min(30, len(features_ready)-1), metric='cosine', random_state=42).fit_transform(features_ready),
    "PCA": PCA(n_components=2).fit_transform(features_ready)
}

# ==============================================================================
# 🛠️ 2. 라벨 파싱 및 빈도수(Count) 계산
# ==============================================================================
MARKER_MAP = {
    "0": "X",  # 0점: 엑스 
    "1": "^",  # 1점: 세모 
    "2": "o",  # 2점: 동그라미 
    "-1": "."  # Rest: 작은 점
}

parsed_data = { "UMAP": [], "t-SNE": [], "PCA": [] }
unique_action_nums = set()

# 💡 [핵심 추가] 각 동작 번호별 개수를 저장할 딕셔너리
action_counts = defaultdict(int)

print("🔍 라벨 파싱 및 분할 매핑 중...")
for i, raw_lbl in enumerate(raw_labels):
    if raw_lbl == "Unknown/Rest":
        action_counts["Rest"] += 1
        for name in embeddings_2d.keys():
            parsed_data[name].append({
                "coord": embeddings_2d[name][i], 
                "action": "Rest", 
                "score": "-1"
            })
        continue
        
    parts = raw_lbl.split('/')
    for part in parts:
        match = re.search(r'GM,?(\d+),(\d+)', part.strip())
        if match:
            action_num = match.group(1)
            score = match.group(2)
            
            unique_action_nums.add(action_num)
            action_counts[action_num] += 1 # 💡 등장 횟수 카운트
            
            for name in embeddings_2d.keys():
                parsed_data[name].append({
                    "coord": embeddings_2d[name][i], 
                    "action": action_num, 
                    "score": score
                })

unique_action_nums = sorted(list(unique_action_nums), key=int)

# ==============================================================================
# 🖌️ 3. 시각화 (색상=동작 번호, 모양=점수, 범례=동작+개수)
# ==============================================================================
for name in embeddings_2d.keys():
    plt.figure(figsize=(16, 12))
    
    cmap = plt.get_cmap('tab20', len(unique_action_nums))
    color_dict = {num: cmap(i) for i, num in enumerate(unique_action_nums)}
    
    plotted_labels = set()
    
    # 1. 배경(Rest) 그리기
    for item in parsed_data[name]:
        if item["action"] == "Rest":
            # 💡 [핵심 수정] 범례에 개수(n) 포함
            lbl_name = f"Unknown/Rest (n={action_counts['Rest']})"
            plt.scatter(item["coord"][0], item["coord"][1], 
                        color='#D3D3D3', marker=MARKER_MAP[item["score"]], s=40, alpha=0.3, zorder=1,
                        label=lbl_name if lbl_name not in plotted_labels else "")
            plotted_labels.add(lbl_name)

    # 2. 실제 행동(Action) 그리기
    for item in parsed_data[name]:
        if item["action"] != "Rest":
            action = item["action"]
            score = item["score"]
            
            # 💡 [핵심 수정] 범례에 개수(n) 포함 (예: "Action 49 (n=13)")
            lbl_name = f"Action {action} (n={action_counts[action]})"
            
            plt.scatter(item["coord"][0], item["coord"][1], 
                        color=color_dict[action], 
                        marker=MARKER_MAP[score], 
                        s=130, 
                        alpha=0.85, 
                        zorder=5, 
                        edgecolors='black' if score == "0" else 'white', 
                        linewidth=0.8,
                        label=lbl_name if lbl_name not in plotted_labels else "")
            plotted_labels.add(lbl_name)

    # ==============================================================================
    # 📝 커스텀 범례 구성
    # ==============================================================================
    plt.title(f"[Experiment B-1] GM Action Clusters ({name}) - Colored by Action, Shaped by Score", fontsize=16, fontweight='bold', pad=15)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    
    import matplotlib.lines as mlines
    score_0 = mlines.Line2D([], [], color='gray', marker='X', linestyle='None', markersize=10, label='Score 0 (Fail)')
    score_1 = mlines.Line2D([], [], color='gray', marker='^', linestyle='None', markersize=10, label='Score 1 (Partial)')
    score_2 = mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=10, label='Score 2 (Pass)')
    
    handles.extend([score_0, score_1, score_2])
    
    plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), shadow=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.4)
    
    out_path = SAVE_DIR / f"clip_clustering_{name.lower()}_by_action.png"
    plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"💾 {name} 시각화 저장 완료: {out_path.name}")

print("\n✨ 모든 분석 및 시각화 작업이 완료되었습니다!")