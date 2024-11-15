
from ultralytics import YOLO
import os


class Model():
    def __init__(self, sufix="") -> None:
        self.model = YOLO(os.path.join('runs', 'segment', f'train{sufix}', 'weights','best.pt'))
    def predict(self, img: str):
        return self.model(img)
