import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.inference import Model
from src.prepare import YOLO_DIR
from src.vis_coco import visualize
import yaml
model = Model()

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


def random_test():
    precision = 4
    ret = []
    with open(os.path.join(YOLO_DIR, 'seg', 'data.yaml')) as stream:
        try:
            datayaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    categories = datayaml['names']
    imgdir = os.path.join(YOLO_DIR, 'seg', 'images', 'test')
    lbldir = os.path.join(YOLO_DIR, 'seg', 'labels', 'test')
    imgfiles = [os.path.splitext(f) for f in os.listdir(os.path.join(YOLO_DIR, 'seg', 'images', 'test'))]
    files = [ (os.path.join(imgdir, f'{base}{ext}'), os.path.join(lbldir, f'{base}.txt')) for base, ext in imgfiles]
    for imgf, lblf in files:
        result = predict(imgf, precision)
        with open(lblf, 'r') as f:
            lns = f.readlines()
        lns = [ln.split(" ") for ln in lns]
        lns = [list(map(float, ln)) for ln in lns]
        lns = [[round(i, precision) for i in ln] for ln in lns]
        lns = [{
            'category': {
                'id':int(ln[0]),
                'name': categories[(int(ln[0]))]
            },
            'segments' : list(zip(ln[1::2], ln[2::2]))
        } for ln in lns]
        ret.append({
            'image_url': imgf,
            'result': result,
            'annotation': lns
        })
    return ret

if __name__ == "__main__":
    rst = random_test()
    for tmp in rst:
        imgf = tmp['image_url']
        result = tmp['result']
        annotation = tmp['annotation']
        visualize(imgf, result, annotation)