from prepare import normalize_cvat, cvat2coco_seg, cvat2coco_kpt, coco_seg2yolo, coco_kpt2yolo
from checkcoco import validate_coco_seg, validate_coco_kpt
if 1== 0:
    normalize_cvat()
elif 1 == 0:
    cvat2coco_seg()
elif 1 == 0:
    cvat2coco_kpt()
elif 1==0:
    validate_coco_seg()
elif 1==0:
    validate_coco_kpt()
elif 1==1:
    coco_kpt2yolo()
elif 1==0:
    coco_seg2yolo()

