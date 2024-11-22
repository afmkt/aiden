import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import time
from ultralytics import YOLO
from src.prepare import YOLO_DIR

import typer


def train(mode='seg', epochs = 100, resume = os.path.join('runs','segment','train','weights','last.pt')):
    if resume is None:
        model = YOLO('yolo11x-seg.pt' if mode == 'seg' else 'yolo11x-pose.pt')
        start = time.time()
        results = model.train(
            data = os.path.join(YOLO_DIR, mode, 'data.yaml')
            , epochs = epochs
            , imgsz = 640
            # pytorch 2.5.1 cause a bug in mps mode
            # , device = 'mps'
            )
        train_end = time.time()
        print(results, start, train_end)
    else:
        model = YOLO(resume)
        start = time.time()
        results = model.train(resume = True)
        train_end = time.time()
        print(results, start, train_end)


app = typer.Typer()
@app.command()
def new(epochs: int = 100):
    train(mode = 'seg', epochs=epochs, resume='')

@app.command()
def resume(last = os.path.join('runs','segment','train','weights','last.pt')):
    train(mode = 'seg', resume=last)



if __name__ == "__main__":
    app()
