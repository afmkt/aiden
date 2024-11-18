
from ultralytics import YOLO
import os
from src.prepare import YOLO_DIR
from typing import Dict
import yaml
import matplotlib.pyplot as plt
import matplotlib
import cv2
import numpy as np
matplotlib.rc('font', family='Hiragino Sans GB')




class Model():
    def __init__(self, model_file = os.path.join('runs', 'segment', 'train', 'weights','best.pt')) -> None:
        self.model = YOLO(model_file)

    def predict(self, img: str, precision = 4):
        result = self.model(img) 
        ret = [
            [{
            'category': {
                'id':int(b.cls[0].item()),
                'name': r.names[int(b.cls[0].item())]
            },
            'width': r.orig_shape[1],
            'height': r.orig_shape[0],
            'confidence': round(b.conf[0].item(), precision),
            'segments': [tuple([round(e[0], precision), round(e[1], precision)]) for e in m.xyn[0]]
        } for b, m in zip(r.boxes, r.masks)] for r in result]
        for r in ret[0]:
            if r['category']['id'] == 0:
                # find the right piece by
                # 1. find the pair of points that has the biggest distance
                # 2. split segments by the index of the pair of points
                # 3. return the piece that has the larger average value of Y
                # r['segments'] = pick_piece(r['segments'])
                segs = r['segments']
                distance = 0.0
                idx1 = None
                idx2 = None
                for i1 in range(len(segs)):
                    for i2 in range(len(segs)):
                        x1, y1 = segs[i1]
                        x2, y2 = segs[i2]
                        d = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
                        if ((idx1 is None) or (idx2 is None)) or (d > distance):
                            distance = max(distance, d)
                            idx1 = i1 
                            idx2 = i2
                idx_s = min(idx1, idx2)
                idx_b = max(idx1, idx2)
                p1 = segs[:idx_s] + segs[idx_b:]
                p2 = segs[idx_s: idx_b]
                avg1 = sum(list(zip(*p1))[1]) / len(p1)
                avg2 = sum(list(zip(*p2))[1]) / len(p2)
                if avg1 > avg2 :
                    r['segments'] = p1
                elif avg1 < avg2:
                    r['segments'] = p2
                elif len(p1) > len(p2):
                    r['segments'] = p1
                else:
                    r['segments'] = p2
        return ret[0]
    
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
MAGENTA = (255, 0, 255)
CYAN = (255, 255, 0)        

def display(image, label= ''):
    while(True):
        cv2.imshow(label, image)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

def plot_result(imgf, result = [], thickness = 2, clrmap = {
    0: RED,
    1: YELLOW,
    2: GREEN,
    3: WHITE
}):
    if isinstance(imgf, str):
        image = cv2.imread(imgf)
    else:
        image = imgf
    height, width, channels = image.shape
    for r in result:
        cat = r['category']
        seg = r['segments']
        x, y = tuple([list(n) for n in zip(*seg)])
        x = [int(i * width) for i in x]
        y = [int(i * height) for i in y]
        pts = np.array(list(map(list, zip(x,y)))).reshape((-1, 1, 2))
        cv2.polylines(image, 
                    [pts], 
                    color=clrmap[cat['id']], 
                    isClosed=False, 
                    thickness=thickness, 
                    lineType=cv2.LINE_8)
    return image

def visualize(imgf, result = [], annotation = []):
    def draw(x, y, cid, alpha, marker):
        match cid:
            case 0:
                plt.plot(x, y, color='red', alpha=alpha, marker=marker)
            case 1:
                plt.plot(x, y, color='yello', alpha=alpha, marker=marker)
            case 2:
                plt.plot(x, y, color='green', alpha=alpha, marker=marker)
            case 3:
                plt.plot(x, y, color='white', alpha=alpha, marker=marker)

    image = cv2.imread(imgf)

    height, width, channels = image.shape    

    if len(result) == 0:
        plt.text(0, 0, f'{os.path.splitext(os.path.basename(imgf))[0]} annotation', color='black')
    elif len(annotation) == 0:
        plt.text(0, 0, f'{os.path.splitext(os.path.basename(imgf))[0]} prediction', color='black')
    plt.imshow(image[..., ::-1])        
    plt.axis('off')
    for r in result:
        cat = r['category']
        seg = r['segments']
        x, y = tuple([list(n) for n in zip(*seg)])
        x = [int(i * width) for i in x]
        y = [int(i * height) for i in y]
        draw(x,y, cat['id'], 1.0, '')
    for a in annotation:
        cat = a['category']
        seg = a['segments']
        x, y = tuple([list(n) for n in zip(*seg)])
        x = [int(i * width) for i in x]
        y = [int(i * height) for i in y]
        draw(x,y, cat['id'], 1.0, '*')
    plt.show()


def load_ann(imgf:str, datayaml: str | Dict | None = os.path.join(YOLO_DIR, 'seg', 'data.yaml'), precision = 4):
    if not isinstance(datayaml, dict):
        if isinstance(datayaml, str):
            with open(datayaml) as stream:
                try:
                    datayaml = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)
        elif datayaml is None:
            with open(os.path.join(YOLO_DIR, 'seg', 'data.yaml')) as stream:
                try:
                    datayaml = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)
        else:
            raise Exception(f'datayaml{datayaml} must be a Dict or a str or None')
    categories = datayaml['names']
    _, filename = os.path.split(imgf)
    base, _ = os.path.splitext(filename)
    for split in ['test', 'train', 'val']:
        lbldir = os.path.join(YOLO_DIR, 'seg', 'labels', split)
        for fname in os.listdir(lbldir):
            b, e = os.path.splitext(fname)
            if b == base:
                with open(os.path.join(lbldir, fname), 'r') as f:
                    lns = f.readlines()
                lns = [ln.split(" ") for ln in lns]
                lns = [list(map(float, ln)) for ln in lns]
                lns = [[round(i, precision) for i in ln] for ln in lns]
                ann =  [{
                    'category': {
                        'id':int(ln[0]),
                        'name': categories[(int(ln[0]))]
                    },
                    'segments' : list(zip(ln[1::2], ln[2::2]))
                } for ln in lns]
                imgdir = os.path.join(YOLO_DIR, 'seg', 'images', split)
                for fname in os.listdir(imgdir):
                    b, e = os.path.splitext(fname)
                    if b == base:
                        return ann, os.path.join(imgdir, fname)

