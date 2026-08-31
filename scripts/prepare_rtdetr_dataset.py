import os
import shutil
import glob
import cv2
import numpy as np
import hashlib
from collections import defaultdict
import yaml

DATA_DIR = r"data\working_dataset"
OUT_DIR = r"data\rtdetr_dataset"

def get_image_phash(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    img = cv2.resize(img, (32, 32))
    dct = cv2.dct(np.float32(img))
    dctlowfreq = dct[0:8, 0:8]
    med = np.median(dctlowfreq)
    diff = dctlowfreq > med
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def main():
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUT_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, split, 'labels'), exist_ok=True)
        
    image_paths = []
    label_paths = {}
    
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(DATA_DIR, split)
        if not os.path.exists(split_dir): continue
        imgs = glob.glob(os.path.join(split_dir, "images", "*.jpg"))
        for img_path in imgs:
            image_paths.append(img_path)
            basename = os.path.basename(img_path)
            name_no_ext = os.path.splitext(basename)[0]
            lbl_path = os.path.join(split_dir, "labels", name_no_ext + ".txt")
            if os.path.exists(lbl_path):
                label_paths[img_path] = lbl_path
                
    phash_dict = defaultdict(list)
    for ip in image_paths:
        ph = get_image_phash(ip)
        if ph is not None:
            phash_dict[ph].append(ip)
            
    unique_phash_groups = list(phash_dict.values())
    np.random.seed(42)
    np.random.shuffle(unique_phash_groups)
    
    total_groups = len(unique_phash_groups)
    train_end = int(total_groups * 0.7)
    val_end = int(total_groups * 0.85)
    
    splits = {
        'train': [f for g in unique_phash_groups[:train_end] for f in g],
        'val': [f for g in unique_phash_groups[train_end:val_end] for f in g],
        'test': [f for g in unique_phash_groups[val_end:] for f in g]
    }
    
    for split_name, files in splits.items():
        for f in files:
            shutil.copy(f, os.path.join(OUT_DIR, split_name, 'images', os.path.basename(f)))
            lbl = label_paths.get(f)
            if lbl:
                shutil.copy(lbl, os.path.join(OUT_DIR, split_name, 'labels', os.path.basename(lbl)))
                
    data_yaml = {
        'path': os.path.abspath(OUT_DIR),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 2,
        'names': ['group', 'solder_point']
    }
    
    with open(os.path.join(OUT_DIR, 'dataset.yaml'), 'w') as f:
        yaml.dump(data_yaml, f)

    print("Dataset prepared at", OUT_DIR)

if __name__ == "__main__":
    main()
