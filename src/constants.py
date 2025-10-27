"""Constants."""

from pathlib import Path

MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
# MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt"
MODEL_CACHE_DIR = Path.home() / ".cache" / "ultralytics"

DETECT_IMAGE = Path("example.jpg")
DEFAULT_POINTS_JSON = Path("points.json")

GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)

CAR_CLASS = 2  # 2 is car class in COCO dataset

TRIANGLE_SIDES = 3
