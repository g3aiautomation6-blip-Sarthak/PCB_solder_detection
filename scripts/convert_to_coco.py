import os
import glob
import json
import cv2
import shutil

YOLO_DIR = r"data\working_dataset"
COCO_DIR = r"data\coco_dataset"

def convert_yolo_to_coco():
    os.makedirs(COCO_DIR, exist_ok=True)
    
    categories = [
        {"id": 0, "name": "group", "supercategory": "none"},
        {"id": 1, "name": "solder_point", "supercategory": "none"}
    ]
    
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(YOLO_DIR, split)
        if not os.path.exists(split_dir): continue
        
        coco_split = "val" if split == "valid" else split
        
        out_img_dir = os.path.join(COCO_DIR, f"{coco_split}2017")
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(os.path.join(COCO_DIR, "annotations"), exist_ok=True)
        
        coco_data = {
            "info": {"description": "Solder Point Dataset", "version": "1.0", "year": 2026},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": categories
        }
        
        img_paths = glob.glob(os.path.join(split_dir, "images", "*.jpg"))
        
        ann_id = 1
        for img_id, img_path in enumerate(img_paths, start=1):
            basename = os.path.basename(img_path)
            shutil.copy(img_path, os.path.join(out_img_dir, basename))
            
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            
            coco_data["images"].append({
                "id": img_id,
                "file_name": basename,
                "height": h,
                "width": w
            })
            
            lbl_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
            if os.path.exists(lbl_path):
                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts: continue
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:])
                        
                        bw_p = bw * w
                        bh_p = bh * h
                        xmin = (cx * w) - (bw_p / 2)
                        ymin = (cy * h) - (bh_p / 2)
                        
                        coco_data["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": cls_id,
                            "bbox": [xmin, ymin, bw_p, bh_p],
                            "area": bw_p * bh_p,
                            "iscrowd": 0
                        })
                        ann_id += 1
                        
        out_ann_path = os.path.join(COCO_DIR, "annotations", f"instances_{coco_split}2017.json")
        with open(out_ann_path, "w") as f:
            json.dump(coco_data, f)
            
    print("COCO dataset created at", COCO_DIR)

if __name__ == "__main__":
    convert_yolo_to_coco()
