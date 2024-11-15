import sys
import os
import shutil
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

def main() -> None:
    from src.prepare import normalize_cvat, cvat2coco_seg, cvat2coco_kpt, coco_seg2yolo, coco_kpt2yolo, yolo_cat_kpt, YOLO_DIR
    from src.vis_coco import validate_coco_seg, validate_coco_kpt
    

    if 1==0:
        normalize_cvat()
    if 1==0:
        cvat2coco_seg()
        cvat2coco_kpt()
    if 1==0:
        validate_coco_seg()
        validate_coco_kpt()
    if 1==1:
        if os.path.isdir(YOLO_DIR):
            shutil.rmtree(YOLO_DIR)
        coco_kpt2yolo()
        coco_seg2yolo()
    if 1==0:
        yolo_cat_kpt()

if __name__ == "__main__":
    main()
