import os
import time
import src.cvat2coco as c
from ultralytics import YOLO

def train(cvatAnn: str = os.path.join('data', 'lowerboneandimplant', 'annotations.xml')):
    train_path = c.cocoAnnFileName(cvatAnn)
    model = YOLO('yolo11x-seg.yaml').load('yolo11x-set.pt')
    start = time.time()
    model.train(data = train_path, epochs = 100, imgsz = 640)
    train_end = time.time()