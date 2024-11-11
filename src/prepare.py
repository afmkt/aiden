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
        return dst, os.path.basename(dst), md5
    return None
def init_dir(dir: str)-> None:
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)

def normalize_cvat(srcdir: str = os.path.join('data', 'lowerboneandImplant'), dstdir : str = os.path.join('data', 'repo')) -> str:
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
            fullpath, basename, md5 = cpfile(imgfile, dstimgdir)
            img.set('name', basename)
            img.set('md5', md5)
            if md5 not in saved_files:
                saved_files[md5] = str(len(saved_files))
            img.set('id', saved_files[md5])
        else:
            should_delete.append(img)
    for n in should_delete:
        root.remove(n)
    cvat.write(os.path.join(dstdir, 'annotations.xml'), encoding='utf8')
    print(f'Normalized CVAT dataset in {dstdir}')
    return dstdir


def cvat2coco(srcdir: str = os.path.join('data', 'repo')):
    def bbox(pnts):
        x, y = zip(*pnts)
        return [min(x), min(y), max(x) - min(x), max(y) - min(y)]        
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
            annotations.append({
                'id': len(annotations),
                'image_id': int(img.get('id')),
                'category_id': categories[plabel]['id'],
                'segmentation': [[r for lst in points for r in lst]],
                'area': Polygon(points).area,
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
            if blabel not in categories:
                categories[blabel] = {
                    'name': blabel,
                    'supercategory': blabel,
                    'id': len(categories)
                }
            annotations.append({
                'id': len(annotations),
                'image_id': int(img.get('id')),
                'category_id': categories[blabel]['id'],
                'area': Polygon(points).area,
                'bbox': bbox([(xtl, ytl),(xtl, ybr),(xbr, ybr),(xbr, ytl)]),
                'iscrowd': 0
            })            
    ann['categories'] = list(categories.values())
    json.dump(ann, open(os.path.join(srcdir, 'coco-annotations.json'), "w"), indent=4)
    

