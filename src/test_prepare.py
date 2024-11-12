from prepare import normalize_cvat, cvat2coco
from checkcoco import validate_coco
srcdir = normalize_cvat()
cvat2coco()
validate_coco()
