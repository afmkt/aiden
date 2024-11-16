import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.inference import Model
from src.prepare import YOLO_DIR
import yaml
import matplotlib.pyplot as plt
import matplotlib
import cv2
from typing import Dict
matplotlib.rc('font', family='Hiragino Sans GB')

model = Model()


def visualize(imgf, result = [], annotation = []):
    def draw(x, y, cid, alpha, marker):
        print(cid)
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
    plt.imshow(image[..., ::-1])
    plt.text(0, 0, os.path.splitext(os.path.basename(imgf))[0], color='black')
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
        draw(x,y, cat['id'], 0.2, '*')
    plt.show()


def predict(imgf, precision = 4):
    result = model.predict(imgf)
    ret = [
        [{
        'category': {
            'id':int(b.cls[0].item()),
            'name': r.names[int(b.cls[0].item())]
        },
        'confidence': round(b.conf[0].item(), precision),
        'segments': [tuple([round(e[0], precision), round(e[1], precision)]) for e in m.xyn[0]]
    } for b, m in zip(r.boxes, r.masks)] for r in result]
    return ret[0]

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
                return [{
                    'category': {
                        'id':int(ln[0]),
                        'name': categories[(int(ln[0]))]
                    },
                    'segments' : list(zip(ln[1::2], ln[2::2]))
                } for ln in lns]

def load_dataset(split = 'test', precision = 4):
    ret = []
    with open(os.path.join(YOLO_DIR, 'seg', 'data.yaml')) as stream:
        try:
            datayaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    imgdir = os.path.join(YOLO_DIR, 'seg', 'images', split)
    lbldir = os.path.join(YOLO_DIR, 'seg', 'labels', split)
    imgfiles = [os.path.splitext(f) for f in os.listdir(os.path.join(YOLO_DIR, 'seg', 'images', split))]
    files = [ (os.path.join(imgdir, f'{base}{ext}'), os.path.join(lbldir, f'{base}.txt')) for base, ext in imgfiles]
    for imgf, lblf in files:
        lns = load_ann(imgf, datayaml, precision = precision)
        ret.append({
            'image_url': imgf,
            'annotation': lns
        })
    return ret

if __name__ == "__main__":
    rst = load_dataset('train', 4)
    for tmp in rst:
        imgf = tmp['image_url']
        annotation = tmp['annotation']
        result = predict(imgf, 4)
        visualize(imgf, [], annotation)