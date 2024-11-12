from prepare import normalize_cvat, cvat2coco_seg, cvat2coco_kpt
from checkcoco import validate_coco_seg, validate_coco_kpt
srcdir = normalize_cvat()
cvat2coco_seg()
cvat2coco_kpt()
if False:
    validate_coco_seg()
else:
    validate_coco_kpt()
