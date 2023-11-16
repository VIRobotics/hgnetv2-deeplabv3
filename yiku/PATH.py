from pathlib import Path
import sys
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

ASSETS = ROOT / 'assets'
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
_WTS_STORAGE_DIR=Path(os.path.expanduser('~'))/ ".lab_model/"
if not os.path.exists(_WTS_STORAGE_DIR):
    os.mkdir(_WTS_STORAGE_DIR)
WTS_STORAGE_DIR=_WTS_STORAGE_DIR