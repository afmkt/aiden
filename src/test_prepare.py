from prepare import normalize_cvat, cvat2coco_seg, cvat2coco_kpt, coco_seg2yolo
from checkcoco import validate_coco_seg, validate_coco_kpt
srcdir = normalize_cvat()
cvat2coco_seg()
cvat2coco_kpt()

if 1==0:
    validate_coco_seg()
elif 1==0:
    validate_coco_kpt()
else:
    coco_seg2yolo()
