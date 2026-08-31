import os
import cv2
import json
import numpy as np

COCO_DIR = r"data\coco_dataset"
VIS_PATH = r"output\visualizations\coco_verification.jpg"

def verify_coco():
    os.makedirs(os.path.dirname(VIS_PATH), exist_ok=True)
    
    ann_file = os.path.join(COCO_DIR, "annotations", "instances_train2017.json")
    if not os.path.exists(ann_file):
        return
        
    with open(ann_file, "r") as f:
        coco_data = json.load(f)
        
    # Get a sample image
    img_info = coco_data["images"][0]
    img_id = img_info["id"]
    img_path = os.path.join(COCO_DIR, "train2017", img_info["file_name"])
    
    img = cv2.imread(img_path)
    if img is None: return
    
    anns = [a for a in coco_data["annotations"] if a["image_id"] == img_id]
    
    for a in anns:
        x, y, w, h = map(int, a["bbox"])
        color = (0, 0, 255) if a["category_id"] == 0 else (0, 255, 0)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        
    cv2.imwrite(VIS_PATH, img)

if __name__ == "__main__":
    verify_coco()
