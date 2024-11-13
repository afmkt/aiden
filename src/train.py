import os
import time
import re
from ultralytics import YOLO
from .prepare import YOLO_DIR
def train(mode):
    if mode == 'seg':
        model = YOLO('yolo11x-seg.pt')
    elif re.search('kpt-[0-9]+', mode) is not None:
        model = YOLO('yolo11x-pose.pt')
    start = time.time()
    results = model.train(data = os.path.join(YOLO_DIR, mode, 'data.yaml'), epochs = 10, imgsz = 640)
    train_end = time.time()
    print(results)