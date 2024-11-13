import xml.etree
import xml.etree.ElementInclude
import xml.etree.ElementTree
from shapely.geometry import Polygon
import json
import xml
import os
import hashlib
import shutil
import cv2
from typing import List, Tuple
import random
from tqdm import tqdm
from pycocotools.coco import COCO
import yaml
from functools import reduce

ORIGINAL_CVAT_DIR = os.path.join('data', 'lowerboneandImplant')
WORKING_DIR = os.path.join('data', 'repo')
YOLO_DIR = os.path.join('data', 'yolo')

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

def normalize_cvat(srcdir: str = ORIGINAL_CVAT_DIR, dstdir : str = WORKING_DIR):
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


def cvat2coco_seg(srcdir: str = WORKING_DIR):
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
    json.dump(ann, open(jsonfile, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
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
    # [x1, y1, x2, y2, ...]
    # we don't use visible flag
    ## 0: Keypoint is not visible.
    ## 1: Keypoint is visible and labeled.
    ## 2: Keypoint is visible but occluded (for instance, partially hidden behind an object or another body part).
    return [r for lst in zip(new_x, new_y) for r in lst]

def cvat2coco_kpt(srcdir: str = WORKING_DIR, kpt_count: int = 34):
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
    json.dump(ann, open(jsonfile, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    print(f"KPT COCO json file at {jsonfile}")



def coco_seg2yolo(srcdir = WORKING_DIR, dstdir = YOLO_DIR, train_ratio = 0.8, val_ratio = 0.1):
    src_imgdir = os.path.join(srcdir, 'images')
    dstdir = os.path.join(dstdir, 'seg')
    datafile = os.path.join(dstdir, 'data.yaml')    
    coco = COCO(os.path.join(srcdir,'coco-seg-annotations.json'))
    imgs = list(coco.imgs.values())
    random.shuffle(imgs)
    train_count = int(train_ratio * len(imgs))
    val_count = int(val_ratio * len(imgs))
    train_imgs = imgs[:train_count]
    val_imgs = imgs[train_count: train_count + val_count]
    test_imgs = imgs[train_count + val_count : ]
    def generate(split, images):
        os.makedirs(os.path.join(dstdir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(dstdir, 'labels', split), exist_ok=True)
        for img in tqdm(images, desc=f'Processing seg dataset for {split}'):
            shutil.copy(os.path.join(src_imgdir, img['file_name']), os.path.join(dstdir, 'images', split, img['file_name']))
            lblfile, ext = os.path.splitext(img['file_name'])
            lblfile = os.path.join(dstdir, 'labels', split, f'{lblfile}.txt')
            img_ids = coco.getImgIds(imgIds=[img['id']])
            ann_ids = coco.getAnnIds(imgIds=img_ids)
            anns = coco.loadAnns(ann_ids)
            imgh = img['height']
            imgw = img['width']
            coco_anns = []
            for ann in anns:
                category_id = ann['category_id']
                bbox = ann['bbox']  # [x, y, width, height]
                minx, miny, w, h = tuple(bbox)
                xcenter = minx + w / 2.0
                ycenter = miny + h / 2.0
                segmentation = ann['segmentation']  
                if len(segmentation) == 1:
                    segmentation = segmentation[0]
                elif len(segmentation) > 1:
                    all_points = []
                    for p in segmentation:
                        all_points.extend(zip(p[::2], p[1::2]))
                    x, y = Polygon(all_points).xy
                    # convert [(x1, y1), (x2, y2) ...] to [x1, y1, x2, y2 ...]
                    segmentation = [item for sublist in zip(x, y) for item in sublist]
                else:
                    raise Exception(f'Invalid number({len(segmentation)}) of polygons in segmentation')
                yolo_line = [
                    category_id, 
                    xcenter / imgw, 
                    ycenter / imgh, 
                    w / imgw, 
                    h / imgh]
                yolo_line.extend([(n / imgw) if idx % 2 == 0 else (n / imgh) for idx, n in enumerate(segmentation)])
                coco_anns.append(yolo_line)
            with open(lblfile, 'w') as file:                    
                file.writelines([" ".join(map(str, a)) for a in coco_anns])
    generate('train', train_imgs)
    generate('val', val_imgs)
    generate('test', test_imgs)
    # write out data.yaml
    category_ids = coco.getCatIds()
    categories = coco.loadCats(category_ids)
    with open(datafile, 'w') as ofile:
        def toobj(a, c):
            a[c['id']] = c['name']
            return a
        yaml.dump({
            'path': os.path.join('yolo', 'seg'),
            'train': os.path.join('images', 'train'),
            'val': os.path.join('images', 'val'),
            'test': os.path.join('images', 'test'),
            'nc': len(categories),
            'names': reduce(toobj  , categories, {})
        }, ofile, explicit_start=True, allow_unicode=True)
            
            
def coco_kpt2yolo(srcdir = WORKING_DIR, dstdir = YOLO_DIR, train_ratio = 0.8, val_ratio = 0.1):
    src_imgdir = os.path.join(srcdir, 'images')
    dstdir = os.path.join(dstdir, 'kpt')
    datafile = os.path.join(dstdir, 'data.yaml')    
    coco = COCO(os.path.join(srcdir,'coco-kpt-annotations.json'))
    imgs = list(coco.imgs.values())
    random.shuffle(imgs)
    train_count = int(train_ratio * len(imgs))
    val_count = int(val_ratio * len(imgs))
    train_imgs = imgs[:train_count]
    val_imgs = imgs[train_count: train_count + val_count]
    test_imgs = imgs[train_count + val_count : ]
    def generate(split, images):
        def cvkpt(tup):
            idx, coord = tup
            if idx % 3 == 0:
                return coord / imgw
            elif idx % 3 == 1:
                return coord / imgh
            elif idx % 3 == 2:
                # the input of this function is encoded in coco
                # 0: Keypoint is not visible.
                # 1: Keypoint is visible and labeled.
                # 2: Keypoint is visible but occluded (for instance, partially hidden behind an object or another body part).
                # the output of this function is in yolo
                # Visibility flag (0 = not labeled, 1 = labeled but not visible, 2 = labeled and visible)
                return 0
        os.makedirs(os.path.join(dstdir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(dstdir, 'labels', split), exist_ok=True)
        kl = None
        for img in tqdm(images, desc=f'Processing kpt dataset for {split}'):
            shutil.copy(os.path.join(src_imgdir, img['file_name']), os.path.join(dstdir, 'images', split, img['file_name']))
            lblfile, ext = os.path.splitext(img['file_name'])
            lblfile = os.path.join(dstdir, 'labels', split, f'{lblfile}.txt')
            img_ids = coco.getImgIds(imgIds=[img['id']])
            ann_ids = coco.getAnnIds(imgIds=img_ids)
            anns = coco.loadAnns(ann_ids)
            imgh = img['height']
            imgw = img['width']
            coco_anns = []
            for ann in anns:
                category_id = ann['category_id']
                bbox = ann['bbox']  # [x, y, width, height]
                minx, miny, w, h = tuple(bbox)
                xcenter = minx + w / 2.0
                ycenter = miny + h / 2.0
                keypoints = ann['keypoints']
                if kl is None:
                    kl = len(keypoints)
                yolo_line = [
                    category_id, 
                    xcenter / imgw, 
                    ycenter / imgh, 
                    w / imgw, 
                    h / imgh]
                yolo_line.extend([(n / imgw) if idx % 2 == 0 else (n / imgh) for idx, n in enumerate(keypoints)])
                coco_anns.append(yolo_line)
            with open(lblfile, 'w') as file:    
                file.writelines([" ".join(map(str, a)) for a in coco_anns])
        return kl
    kpt_length = generate('train', train_imgs)
    generate('val', val_imgs)
    generate('test', test_imgs)
    # write out data.yaml
    category_ids = coco.getCatIds()
    categories = coco.loadCats(category_ids)
    with open(datafile, 'w') as ofile:
        def toobj(a, c):
            a[c['id']] = c['name']
            return a
        yaml.dump({
            'path': os.path.join('yolo', 'kpt'),
            'train': os.path.join('images', 'train'),
            'val': os.path.join('images', 'val'),
            'test': os.path.join('images', 'test'),
            'kpt_shape': [kpt_length / 2, 2],
            'nc': len(categories),
            'names': reduce(toobj  , categories, {})
        }, ofile, explicit_start=True, allow_unicode=True)


def yolo_cat_kpt(dir = YOLO_DIR):
    with open(os.path.join(dir, 'kpt', 'data.yaml')) as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    for cid, cstr in data['names'].items():
        with open(os.path.join(YOLO_DIR, f'kpt-{cid}', 'data.yaml'), 'w') as ofile:
            yaml.dump({
                'path': os.path.join('yolo', 'kpt'),
                'train': os.path.join('images', 'train'),
                'val': os.path.join('images', 'val'),
                'test': os.path.join('images', 'test'),
                'kpt_shape': data['kpt_shape'],
                'nc': 1,
                'names': {
                    f'{cid}': cstr
                }
            }, ofile, explicit_start=True, allow_unicode=True)
        for split in ['train' 'val', 'test']:
            srcimgdir = os.path.join(dir, 'kpt', 'images', split)
            srcimgnames = os.listdir(srcimgdir)
            srclbldir = os.path.join(dir, 'kpt', 'labels', split)
            srclblnames = os.listdir(srclbldir)
            imgdir = os.path.join(YOLO_DIR, f'kpt-{cid}', 'images', split)
            lbldir = os.path.join(YOLO_DIR, f'kpt-{cid}', 'labels', split)
            for lblfile in srclblnames:
                with open(os.path.join(srclbldir, lblfile), 'r') as f:
                    lns = filter(lambda ln: ln.split(' ')[0].strip() == str(cid), f)
                if len(lns) > 0:
                    with open(os.path.join(lbldir, lblfile), 'w') as file:    
                        file.writelines(lns)
                    base, ext = os.path.splitext(lblfile)
                    for n in srcimgnames:
                        b, _ = os.splitext(os.path.basename(n))
                        if b == base:
                            shutil.copy(os.path.join(srcimgdir, n), os.path.join(imgdir, n))
                            break
            
        

            
            
            
            
            
            datayaml = os.path.join(YOLO_DIR, f'kpt-{cid}', 'data.yaml')

        

            pass

