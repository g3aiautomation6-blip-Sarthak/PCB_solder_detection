import json
import os
import shutil

# 1. Preserve Baseline-A
src_model_dir = "models/rtdetrv2_train_baseline"
dst_model_dir = "models/baseline_A_two_class"
if os.path.exists(src_model_dir) and not os.path.exists(dst_model_dir):
    shutil.copytree(src_model_dir, dst_model_dir)
    print(f"Copied {src_model_dir} to {dst_model_dir}")

# 2. Create Solder-Only dataset annotations
os.makedirs("data/coco_dataset_solder_only/annotations", exist_ok=True)
os.makedirs("output/reports", exist_ok=True)

splits = ["train2017", "val2017", "test2017"]
stats = {}

for split in splits:
    in_file = f"data/coco_dataset/annotations/instances_{split}.json"
    out_file = f"data/coco_dataset_solder_only/annotations/instances_{split}.json"
    
    with open(in_file, 'r') as f:
        data = json.load(f)
        
    original_ann_count = len(data['annotations'])
    
    # Filter out group (id 0) and remap solder_point (id 1 -> 0)
    new_anns = []
    points_per_img = {}
    
    for ann in data['annotations']:
        if ann['category_id'] == 1: # solder_point
            ann['category_id'] = 0
            new_anns.append(ann)
            points_per_img[ann['image_id']] = points_per_img.get(ann['image_id'], 0) + 1
            
    data['annotations'] = new_anns
    data['categories'] = [{'id': 0, 'name': 'solder_point', 'supercategory': 'none'}]
    
    with open(out_file, 'w') as f:
        json.dump(data, f)
        
    stats[split] = {
        'images': len(data['images']),
        'original_anns': original_ann_count,
        'solder_anns': len(new_anns),
        'point_distribution': {}
    }
    
    for img_id, count in points_per_img.items():
        stats[split]['point_distribution'][count] = stats[split]['point_distribution'].get(count, 0) + 1

# Generate report
with open("output/reports/solder_only_dataset_report.txt", "w") as f:
    f.write("SOLDER-ONLY DATASET VERIFICATION REPORT\n")
    f.write("=======================================\n")
    for split, s in stats.items():
        f.write(f"\nSplit: {split}\n")
        f.write(f"  Images: {s['images']}\n")
        f.write(f"  Original Annotations: {s['original_anns']}\n")
        f.write(f"  Solder Annotations (Retained): {s['solder_anns']}\n")
        f.write(f"  Group Annotations (Removed): {s['original_anns'] - s['solder_anns']}\n")
        f.write(f"  Point Count Distribution (Images with N points):\n")
        for count in sorted(s['point_distribution'].keys(), reverse=True):
            f.write(f"    {count} points: {s['point_distribution'][count]} images\n")

print("Dataset preparation and verification completed.")
