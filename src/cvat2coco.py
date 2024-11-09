import xml.etree
import xml.etree.ElementInclude
import xml.etree.ElementTree
from shapely.geometry import Polygon
import json
import xml
import os
import os.path as path
import hashlib
import shutil
import numpy as np
import cv2
from typing import List, Tuple
import random

class Repo:
    def is_train(self):
        return random.random() < self.train_ratio
    def __init__(self, basedir: str, train_ratio: float):
        if train_ratio >1 or train_ratio < 0:
            raise Exception(f"training/validation ratio({train_ratio}) should lies in (0, 1)")
        self.train_ratio = train_ratio
        self.segbasedir = os.path.join(basedir, 'seg')
        self.segimagestrain = os.path.join(self.segbasedir, 'images', 'train')
        self.seglabelstrain = os.path.join(self.segbasedir, 'labels', 'train')
        self.segimagesval = os.path.join(self.segbasedir, 'images', 'val')
        self.seglabelsval = os.path.join(self.segbasedir, 'labels', 'val')

        os.makedirs(self.segimagestrain, exist_ok=True)
        os.makedirs(self.seglabelstrain, exist_ok=True)
        os.makedirs(self.segimagesval, exist_ok=True)
        os.makedirs(self.seglabelsval, exist_ok=True)

        self.detbasedir = os.path.join(basedir, 'det')
        self.detimagestrain = os.path.join(self.detbasedir, 'images', 'train')
        self.detlabelstrain = os.path.join(self.detbasedir, 'labels', 'train')
        self.detimagesval = os.path.join(self.detbasedir, 'images', 'val')
        self.detlabelsval = os.path.join(self.detbasedir, 'labels', 'val')

        os.makedirs(self.detimagestrain, exist_ok=True)
        os.makedirs(self.detlabelstrain, exist_ok=True)
        os.makedirs(self.detimagesval, exist_ok=True)
        os.makedirs(self.detlabelsval, exist_ok=True)



    def commitRectangle(filename: str, width: int, height: int, points: Tuple[int, int, int, int])->None:
        pass
    def commitPolyline(filename: str, width: int, height: int, points: List[Tuple[int, int]])->None:
        pass

def drawPolyline(filename: str, width: int, height: int, points: List[Tuple[int, int]])->None:
    image = np.zeros((width, height, 3), dtype=np.uint8)
    points = np.array(points, np.int32)
    points = points.reshape((-1, 1, 2))
    cv2.fillPoly()
    cv2.polylines(image, [points], isClosed=True, color=(255, 255, 255))
    cv2.imwrite(f'mask.{filename}', image)
    


def commit(basedir: str, imgfile: str):
    absimgfile = os.path.join(basedir, imgfile)
    if os.path.exists(absimgfile):
        dstdir = os.path.join(basedir, 'repo')
        os.makedirs(dstdir, exist_ok=True)
        ext = os.path.splitext(imgfile)[1]
        md5 = hashlib.md5(open(absimgfile, 'rb').read()).hexdigest()
        if not os.path.exists(os.path.join(dstdir, f'{md5}{ext}')):
            shutil.copy(absimgfile, os.path.join(dstdir, f'{md5}{ext}'))
        return os.path.join('repo', f'{md5}{ext}'), md5
    else:
        return imgfile

def cocoAnnFileName(cvatAnn:str):
    absp = path.abspath(path.dirname(cvatAnn))
    b = path.basename(cvatAnn)
    b = path.splitext(b)[0]
    os.makedirs(path.join(absp, 'coco'), exist_ok=True)
    p = path.join(absp,'coco', f'coco-{b}.json')
    return p

def cvat2coco(cvatAnn: str, ignoreNonExistingImage = True):
    def bbox(pnts):
        x, y = zip(*pnts)
        return [min(x), min(y), max(x) - min(x), max(y) - min(y)]    
    absp = path.abspath(path.dirname(cvatAnn))
    cvat = xml.etree.ElementTree.parse(cvatAnn)
    root = cvat.getroot()
    ann = {
        'info':{
            'year': 2024,
            'version': '0.0.1',
            'description':'Dentistry data',
            'contributor':'',
            'url':'',
            'date_created': '2024-05-20'
            },
        'images':[],
        'annotations':[],
        'categories':[]
    }
    categories ={}
    images = ann['images']
    imginrepo = {}
    annotations = ann['annotations']
    for img in root.iter('image'):
        imgname = img.get('name')
        if ignoreNonExistingImage and (not os.path.exists(os.path.join(absp, 'images', imgname))):
            continue
        imgwidth = img.get('width')
        imgheight = img.get('height')        
        # fill images
        repoimgname, md5 = commit(absp, path.join('images',imgname))
        if not md5 in imginrepo:
            imginrepo[md5] = len(images)
            images.append({
                'file_name': repoimgname,
                'width': int(imgwidth),
                'height': int(imgheight),
                'id': imginrepo[md5]
            })
        imgid = imginrepo[md5]
        for polyline in img.iter('polyline'):
            plabel = polyline.get('label')
            points = list(map(lambda p: (float(p.split(',')[0]), float(p.split(',')[1])), polyline.get('points').split(';')))
            # populate categories
            if plabel in categories:
                k = categories[plabel]['keypoints']
                if len(points) > len(k):
                    categories[plabel]['keypoints'] = list(map(lambda i: str(i), range(len(points))))
                    categories[plabel]['skeleton'] = list(zip(range(len(points)), range(len(points))[1:]))
            else:               
                categories[plabel] = {
                    'name': plabel,
                    'supercategory': plabel,
                    'id': len(categories),
                    'keypoints': list(map(lambda i: str(i), range(len(points)))),
                    'skeleton': list(zip(range(len(points)), range(len(points))[1:]))
                }
            # populate annotations
            annotations.append({
                'id': len(annotations),
                'image_id': imgid,
                'category_id': categories[plabel]['id'],
                'segmentation': [[r for lst in points for r in lst]],
                'area': Polygon(points).area,
                # 'bbox': bbox(points),
                'iscrowd': 0,
                'keypoints': [r for lst in map(lambda pnt: list(pnt) + [2], points) for r in lst],
                'num_keypoints': len(points)
            })

        for box in img.iter('box'):
            blabel = box.get('label')
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            # populate categories
            if blabel not in categories:
                categories[blabel] = {
                    'name': blabel,
                    'supercategory': blabel,
                    'id': len(categories)
                }
            # populate annotations
            annotations.append({
                'id': len(annotations),
                'image_id': imgid,
                'category_id': categories[blabel]['id'],
                'area': Polygon(points).area,
                'bbox': bbox([(xtl, ytl),(xtl, ybr),(xbr, ybr),(xbr, ytl)]),
                'iscrowd': 0
            })            
    ann['categories'] = list(categories.values())
    p = cocoAnnFileName(cvatAnn)
    json.dump(ann, open(p, "w"), indent=4)
    return p




if __name__ == "__main__":
    cvat2coco('./data/lowerboneandimplant/annotations.xml')