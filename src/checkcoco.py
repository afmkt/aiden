from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import os
import cv2
import matplotlib
matplotlib.rc('font', family='Hiragino Sans GB')

def validate_image_id(coco):
    image_ids = {img['id'] for img in coco.imgs.values()}
    for ann in coco.anns.values():
        if ann['image_id'] not in image_ids:
            raise Exception(f"Invalid image_id {ann['image_id']} in annotations.")


def validate_seg_coords(coco):
    for ann in coco.anns.values():
        img = coco.imgs[ann['image_id']]
        width, height = img['width'], img['height']
        if 'segmentation' in ann:
            if isinstance(ann['segmentation'], list):  # Polygon format
                for segment in ann['segmentation']:
                    for i in range(0, len(segment), 2):
                        x, y = segment[i], segment[i + 1]
                        if x < 0 or x >= width or y < 0 or y >= height:
                            raise Exception(f"Out-of-bounds coordinate ({x}, {y}) in annotation {ann['id']}")
            elif isinstance(ann['segmentation'], dict):  # RLE format
                # If using RLE, check if there's an alignment issue with image size
                # This check may vary depending on your RLE encoding.
                raise Exception('RLE format')
                pass  # RLE validation could depend on your encoding method.
        else:
            print(ann)
            print('********************************************')

def validate_category_id(coco):
    category_ids = {cat['id'] for cat in coco.cats.values()}
    for ann in coco.anns.values():
        if ann['category_id'] not in category_ids:
            raise Exception(f"Invalid category_id {ann['category_id']} in annotation {ann['id']}")


def validate_coco(annfile: str = 'data/repo/coco-annotations.json', img_dir = 'data/repo/images'):
    coco = COCO(annfile)
    validate_category_id(coco)
    validate_image_id(coco)
    validate_seg_coords(coco)
    for img in coco.imgs.values():
        imgfile = f"{img_dir}/{img['file_name']}"
        image = cv2.imread(imgfile)
        plt.imshow(image[..., ::-1])
        plt.axis('off')

        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)        
        for ann in anns:
            # Get category name
            catId = ann['category_id']
            catName = coco.loadCats(ids=[catId])[0]['name']

            # Get segmentation mask
            mask = coco.annToMask(ann)

            # Plot mask with color
            plt.imshow(mask, alpha=0.5)

            # Add label
            bbox = ann['bbox']
            x, y, w, h = bbox
            plt.text(x, y, catName, color='white')
        plt.show()

