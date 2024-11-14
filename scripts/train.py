import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import time
import re
from ultralytics import YOLO
from src.prepare import YOLO_DIR
def train(mode, epochs = 10):
    if mode == 'seg':
        model = YOLO('yolo11x-seg.pt')
    elif re.search('kpt-[0-9]+', mode) is not None:
        model = YOLO('yolo11x-pose.pt')
    start = time.time()
    results = model.train(
        data = os.path.join(YOLO_DIR, mode, 'data.yaml')
        , epochs = epochs
        , imgsz = 640
        # pytorch 2.5.1 cause a bug in mps mode
        # , device = 'mps'
        )
    train_end = time.time()
    print(results)


if __name__ == "__main__":
    train('seg', 10)
