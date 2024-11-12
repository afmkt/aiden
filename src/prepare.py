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
from tqdm import tqdm



def cpfile(src: str, dst: str)-> Tuple[str, str, str] | None:
    if os.path.isfile(src) and os.path.isdir(dst):
        ext = os.path.splitext(src)[1]
        md5 = hashlib.md5(open(src, 'rb').read()).hexdigest()
        dst = os.path.join(dst, f'{md5}{ext}')
        shutil.copy(src, dst)
        return os.path.dirname(dst), os.path.basename(dst), md5
    return None
def init_dir(dir: str)-> None:
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)

def normalize_cvat(srcdir: str = os.path.join('data', 'lowerboneandImplant'), dstdir : str = os.path.join('data', 'repo')):
    print('''Normalize CVAT dataset bysaving 
1. Save images with md5 checksum as file name
2. Removing non-existing image from XML annotations.xml''')
    xmlfile = os.path.join(srcdir, 'annotations.xml')
    cvat = xml.etree.ElementTree.parse(xmlfile, parser=xml.etree.ElementTree.XMLParser(encoding="utf-8"))
    root = cvat.getroot()
    dstimgdir = os.path.join(dstdir, 'images')
    init_dir(dstimgdir)
    should_delete = []      
    saved_files = {}
    for img in tqdm(root.iter('image'), total=len(root.findall('image'))):
        imgfile = os.path.join(srcdir, 'images', img.get('name'))
        if os.path.isfile(imgfile):
            image = cv2.imread(imgfile)
            h, w = image.shape[0], image.shape[1]
            if h != int(img.get('height')):
                img.set('height', h)
            if w != int(img.get('width')):
                img.set('width', w)
            dir, basename, md5 = cpfile(imgfile, dstimgdir)
            img.set('md5', md5)
            if md5 not in saved_files:
                saved_files[md5] = str(len(saved_files))
            img.set('id', saved_files[md5])
            fname = f'{str(saved_files[md5]).zfill(4)}-{basename}' 
            img.set('name', fname)
            os.rename(os.path.join(dir, basename), os.path.join(dir, fname))
        else:
            should_delete.append(img)
    for n in should_delete:
        root.remove(n)
    cvat.write(os.path.join(dstdir, 'annotations.xml'), encoding='utf8')
    print(f'Normalized CVAT dataset in {dstdir}')


def cvat2coco_seg(srcdir: str = os.path.join('data', 'repo')):
    xmlfile = os.path.join(srcdir, 'annotations.xml')
    cvat = xml.etree.ElementTree.parse(xmlfile, parser=xml.etree.ElementTree.XMLParser(encoding="utf-8"))
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
    annotations = ann['annotations']

    for img in root.iter('image'):
        imgname = img.get('name')
        imgwidth = img.get('width')
        imgheight = img.get('height')  
        # fill images
        images.append({
            'file_name': imgname,
            'width': int(imgwidth),
            'height': int(imgheight),
            'id': int(img.get('id'))
        })
        for polyline in img.iter('polyline'):
            plabel = polyline.get('label')
            points = list(map(lambda p: (float(p.split(',')[0]), float(p.split(',')[1])), polyline.get('points').split(';')))
            if plabel not in categories:
                categories[plabel] = {
                    'name': plabel,
                    'supercategory': plabel,
                    'id': len(categories)
                }
            minx, miny, maxx, maxy = Polygon(points).bounds
            annotations.append({
                'id': len(annotations),
                'image_id': int(img.get('id')),
                'category_id': categories[plabel]['id'],
                'segmentation': [[r for lst in points for r in lst]],
                'area': Polygon(points).area,
                'bbox': [minx, miny, maxx - minx, maxy - miny],
                'iscrowd': 0,
            })

        for box in img.iter('box'):
            blabel = box.get('label')
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            points = [(xtl, ytl), (xtl, ybr), (xbr, ybr), (xbr, ytl)]
            if blabel not in categories:
                categories[blabel] = {
                    'name': blabel,
                    'supercategory': blabel,
                    'id': len(categories)
                }
            minx, miny, maxx, maxy = min(xtl, xbr), min(ytl, ybr), max(xtl, xbr), max(ytl, ybr)
            annotations.append({
                'id': len(annotations),
                'image_id': int(img.get('id')),
                'category_id': categories[blabel]['id'],
                'segmentation': [[r for lst in points for r in lst]],
                'area': Polygon(points).area,
                'bbox': [minx, miny, maxx - minx, maxy - miny],
                'iscrowd': 0
            })            
    ann['categories'] = list(categories.values())
    jsonfile = os.path.join(srcdir, 'coco-seg-annotations.json')
    json.dump(ann, open(jsonfile, "w"), indent=4)
    print(f"SEG COCO json file at {jsonfile}")
    
def seg2kpt(seg: List[Tuple[float, float]], count: int)->List[float]:
    from scipy.interpolate import CubicSpline
    import numpy as np
    pnts = np.array(seg)
    i = np.arange(len(pnts))
    x = pnts[:, 0]
    y = pnts[:, 1]
    cs_x = CubicSpline(i, x)
    cs_y = CubicSpline(i, y)
    new_i = np.linspace(0, len(pnts) - 1, count)
    new_x = cs_x(new_i) 
    new_y = cs_y(new_i)
    # [x1, y1, 2, x2, y2, 2, ...]
    return [r for lst in [(*t, 2) for t in zip(new_x, new_y)] for r in lst]

def cvat2coco_kpt(srcdir: str = os.path.join('data', 'repo'), kpt_count: int = 34):
    xmlfile = os.path.join(srcdir, 'annotations.xml')
    cvat = xml.etree.ElementTree.parse(xmlfile, parser=xml.etree.ElementTree.XMLParser(encoding="utf-8"))
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
    annotations = ann['annotations']

    for img in root.iter('image'):
        imgname = img.get('name')
        imgwidth = img.get('width')
        imgheight = img.get('height')  
        # fill images
        images.append({
            'file_name': imgname,
            'width': int(imgwidth),
            'height': int(imgheight),
            'id': int(img.get('id'))
        })
        for polyline in img.iter('polyline'):
            plabel = polyline.get('label')
            points : List[Tuple[float, float]] = list(map(lambda p: (float(p.split(',')[0]), float(p.split(',')[1])), polyline.get('points').split(';')))
            if plabel not in categories:
                categories[plabel] = {
                    'name': plabel,
                    'supercategory': plabel,
                    'id': len(categories),
                    'skeleton': list(map(list, zip(range(1, kpt_count), range(2, kpt_count + 1))))
                }
            minx, miny, maxx, maxy = Polygon(points).bounds
            annotations.append({
                'id': len(annotations),
                'image_id': int(img.get('id')),
                'category_id': categories[plabel]['id'],
                'keypoints': seg2kpt(points, kpt_count),
                'num_keypoints': kpt_count, 
                'area': Polygon(points).area,
                'bbox': [minx, miny, maxx - minx, maxy - miny],
                'iscrowd': 0,
            })

        for box in img.iter('box'):
            blabel = box.get('label')
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            points = [(xtl, ytl), (xtl, ybr), (xbr, ybr), (xbr, ytl)]
            if blabel not in categories:
                categories[blabel] = {
                    'name': blabel,
                    'supercategory': blabel,
                    'id': len(categories),
                    'skeleton': list(map(list, zip(range(1, 4), range(2, 4 + 1))))
                }
            minx, miny, maxx, maxy = min(xtl, xbr), min(ytl, ybr), max(xtl, xbr), max(ytl, ybr)
            annotations.append({
                'id': len(annotations),
                'image_id': int(img.get('id')),
                'category_id': categories[blabel]['id'],
                'keypoints': seg2kpt(points, kpt_count),
                'num_keypoints': 4, 
                'area': Polygon(points).area,
                'bbox': [minx, miny, maxx - minx, maxy - miny],
                'iscrowd': 0
            })            
    ann['categories'] = list(categories.values())
    jsonfile = os.path.join(srcdir, 'coco-kpt-annotations.json')
    json.dump(ann, open(jsonfile, "w"), indent=4)
    print(f"KPT COCO json file at {jsonfile}")



def coco_seg2yolo(srcdir = os.path.join('data', 'repo'), dstdir = os.path.join('data', 'yolo', 'label')):
    import supervision as sv
    if not os.path.isdir(dstdir):
        os.makedirs(dstdir, exist_ok=True)
    ds = sv.DetectionDataset.from_coco(
        images_directory_path = os.path.join(srcdir, 'images'), 
        annotations_path = os.path.join(srcdir,'coco-seg-annotations.json'))
    ds.as_yolo(
        annotations_directory_path = os.path.join(dstdir,'labels'),
        data_yaml_path = os.path.join(dstdir, 'data.yaml'))
    