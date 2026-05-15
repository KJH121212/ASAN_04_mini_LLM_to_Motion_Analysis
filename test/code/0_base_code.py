import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

print(f"📁 프로젝트 루트 경로: {PROJECT_ROOT}")