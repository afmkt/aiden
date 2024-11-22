import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.inference import Model, load_ann, visualize, plot_result, display
from src.prepare import YOLO_DIR
import yaml
import typer
from enum import Enum
class Category(Enum):
    TEST='test'
    TRAIN='train'
    VAL='val'






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
        lns, _ = load_ann(imgf, datayaml, precision = precision)
        ret.append({
            'image_url': imgf,
            'annotation': lns
        })
    return ret




app = typer.Typer()

model = Model()

@app.command()
def image(name: str):
    ann, imgf = load_ann(name)
    visualize(imgf, [], ann)
    r = model.predict(imgf)
    visualize(imgf, r)

@app.command()
def random(category: Category):
    rst = load_dataset(category.value, 4)
    for tmp in rst:
        imgf = tmp['image_url']
        annotation = tmp['annotation']
        result, width, height = model.predict(imgf, 4)
        visualize(imgf, [], annotation)
        # visualize(imgf, result)
        img = plot_result(imgf, result)
        display(img, f'{imgf} prediction')




if __name__ == "__main__":
    app()