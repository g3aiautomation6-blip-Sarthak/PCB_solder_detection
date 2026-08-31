import os
import cv2
import glob
import json
import hashlib
from collections import defaultdict
import numpy as np

DATA_DIR = r"data\working_dataset"
REPORT_PATH = r"output\reports\audit_report.txt"
VISUALIZATIONS_DIR = r"output\visualizations"

def hash_image(image_path):
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def get_image_phash(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (32, 32))
    dct = cv2.dct(np.float32(img))
    dctlowfreq = dct[0:8, 0:8]
    med = np.median(dctlowfreq)
    diff = dctlowfreq > med
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def run_audit():
    total_images = 0
    total_labels = 0
    resolutions = defaultdict(int)
    class_counts = {0: 0, 1: 0}
    other_classes = defaultdict(int)
    
    exact_12_pts = 0
    exact_4_grps = 0
    
    missing_labels = []
    extra_labels = []
    malformed_labels = []
    suspicious_annotations = []
    
    image_names = set()
    duplicate_names = []
    
    image_paths = []
    label_paths = {}
    
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(DATA_DIR, split)
        if not os.path.exists(split_dir): continue
        
        imgs = glob.glob(os.path.join(split_dir, "images", "*.*"))
        lbls = glob.glob(os.path.join(split_dir, "labels", "*.txt"))
        
        for img_path in imgs:
            if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')): continue
            total_images += 1
            image_paths.append(img_path)
            
            basename = os.path.basename(img_path)
            name_no_ext = os.path.splitext(basename)[0]
            if name_no_ext in image_names:
                duplicate_names.append(name_no_ext)
            image_names.add(name_no_ext)
            
            # Read image resolution
            img = cv2.imread(img_path)
            if img is not None:
                h, w, _ = img.shape
                resolutions[f"{w}x{h}"] += 1
            
            lbl_path = os.path.join(split_dir, "labels", name_no_ext + ".txt")
            if os.path.exists(lbl_path):
                total_labels += 1
                label_paths[img_path] = lbl_path
                
                with open(lbl_path, "r") as f:
                    lines = f.readlines()
                
                pts_count = 0
                grps_count = 0
                
                for line_idx, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        malformed_labels.append(f"{lbl_path} line {line_idx+1}: {line.strip()}")
                        continue
                    
                    try:
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:])
                        
                        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= bw <= 1 and 0 <= bh <= 1):
                            suspicious_annotations.append(f"{lbl_path} line {line_idx+1}: Out of range bbox")
                            
                        if cls_id == 0:
                            grps_count += 1
                            class_counts[0] += 1
                        elif cls_id == 1:
                            pts_count += 1
                            class_counts[1] += 1
                        else:
                            other_classes[cls_id] += 1
                    except ValueError:
                        malformed_labels.append(f"{lbl_path} line {line_idx+1}: Not numeric")
                        
                if pts_count == 12:
                    exact_12_pts += 1
                if grps_count == 4:
                    exact_4_grps += 1
                    
                if pts_count != 12 or grps_count != 4:
                    extra_labels.append(f"{lbl_path}: {pts_count} pts, {grps_count} grps")
            else:
                missing_labels.append(img_path)
                
    # Detect duplicates
    hash_dict = defaultdict(list)
    phash_dict = defaultdict(list)
    
    for ip in image_paths:
        h = hash_image(ip)
        hash_dict[h].append(ip)
        
        ph = get_image_phash(ip)
        if ph is not None:
            phash_dict[ph].append(ip)
            
    exact_duplicates = {k: v for k, v in hash_dict.items() if len(v) > 1}
    near_duplicates = {k: v for k, v in phash_dict.items() if len(v) > 1}
    
    # Generate report
    with open(REPORT_PATH, "w") as f:
        f.write("=== DATASET AUDIT REPORT ===\n")
        f.write(f"Total Images: {total_images}\n")
        f.write(f"Total Label Files: {total_labels}\n")
        f.write("\nImage Resolutions:\n")
        for res, count in resolutions.items():
            f.write(f"  {res}: {count}\n")
        
        f.write(f"\nClass Counts:\n")
        f.write(f"  Class 0 (Group): {class_counts[0]}\n")
        f.write(f"  Class 1 (Solder Point): {class_counts[1]}\n")
        if other_classes:
            f.write(f"  Other Classes: {dict(other_classes)}\n")
            
        f.write(f"\nImages with exactly 12 solder-point annotations: {exact_12_pts}\n")
        f.write(f"Images with exactly 4 group annotations: {exact_4_grps}\n")
        
        f.write(f"\nMissing Labels: {len(missing_labels)}\n")
        for m in missing_labels[:10]: f.write(f"  {m}\n")
        
        f.write(f"\nMalformed Labels: {len(malformed_labels)}\n")
        for m in malformed_labels[:10]: f.write(f"  {m}\n")
        
        f.write(f"\nImages with missing/extra annotations (expected 12 pts, 4 grps): {len(extra_labels)}\n")
        for e in extra_labels[:10]: f.write(f"  {e}\n")
        
        f.write(f"\nSuspicious Annotations (e.g. out of bounds): {len(suspicious_annotations)}\n")
        for s in suspicious_annotations[:10]: f.write(f"  {s}\n")
        
        f.write(f"\nDuplicate Image Names: {len(duplicate_names)}\n")
        
        f.write("\n=== DUPLICATE / NEAR-DUPLICATE FINDINGS ===\n")
        f.write(f"Exact Content Duplicates (MD5): {len(exact_duplicates)}\n")
        for h, files in exact_duplicates.items():
            f.write(f"  Hash {h}: {files}\n")
            
        f.write(f"\nNear Duplicates (pHash): {len(near_duplicates)}\n")
        for h, files in near_duplicates.items():
            f.write(f"  pHash {h}: {len(files)} files\n")
            for file in files:
                f.write(f"    {file}\n")
                
    # Visualize a sample
    if len(image_paths) > 0:
        sample_img_path = image_paths[0]
        sample_lbl_path = label_paths.get(sample_img_path)
        if sample_lbl_path:
            img = cv2.imread(sample_img_path)
            h, w, _ = img.shape
            with open(sample_lbl_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])
                    
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    
                    color = (0, 0, 255) if cls_id == 0 else (0, 255, 0)
                    label = "Group(0)" if cls_id == 0 else "Point(1)"
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            cv2.imwrite(os.path.join(VISUALIZATIONS_DIR, "sample_visualization.jpg"), img)
            
    # Proposed train/val/test split
    # Simple split avoiding exact pHash collisions across splits
    unique_phash_groups = list(phash_dict.values())
    np.random.seed(42)
    np.random.shuffle(unique_phash_groups)
    
    total_groups = len(unique_phash_groups)
    train_end = int(total_groups * 0.7)
    val_end = int(total_groups * 0.85)
    
    train_groups = unique_phash_groups[:train_end]
    val_groups = unique_phash_groups[train_end:val_end]
    test_groups = unique_phash_groups[val_end:]
    
    train_files = [f for g in train_groups for f in g]
    val_files = [f for g in val_groups for f in g]
    test_files = [f for g in test_groups for f in g]
    
    with open(REPORT_PATH, "a") as f:
        f.write("\n=== PROPOSED TRAIN/VAL/TEST SPLIT ===\n")
        f.write("Split strategy: Grouped by near-duplicates (pHash) to avoid leakage.\n")
        f.write(f"Total Unique Image Groups: {total_groups}\n")
        f.write(f"Train: {len(train_files)} images ({len(train_files)/total_images*100:.1f}%)\n")
        f.write(f"Val: {len(val_files)} images ({len(val_files)/total_images*100:.1f}%)\n")
        f.write(f"Test: {len(test_files)} images ({len(test_files)/total_images*100:.1f}%)\n")

if __name__ == "__main__":
    run_audit()
