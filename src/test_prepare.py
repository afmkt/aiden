from prepare import normalize_cvat, cvat2coco_seg, cvat2coco_kpt, coco_seg2yolo, coco_kpt2yolo, yolo_cat_kpt
from checkcoco import validate_coco_seg, validate_coco_kpt
if 1== 0:
    normalize_cvat()
if 1==0:
    cvat2coco_seg()
    cvat2coco_kpt()
if 1==0:
    validate_coco_seg()
    validate_coco_kpt()
if 1==1:
    coco_kpt2yolo()
    coco_seg2yolo()
if 1==1:
    yolo_cat_kpt()

