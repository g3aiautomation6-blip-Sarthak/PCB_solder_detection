import os
import glob
import torch
from ultralytics import RTDETR
import numpy as np

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    b1_x1, b1_y1 = x1 - w1/2, y1 - h1/2
    b1_x2, b1_y2 = x1 + w1/2, y1 + h1/2
    b2_x1, b2_y1 = x2 - w2/2, y2 - h2/2
    b2_x2, b2_y2 = x2 + w2/2, y2 + h2/2
    
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    b1_area = w1 * h1
    b2_area = w2 * h2
    iou = inter_area / float(b1_area + b2_area - inter_area + 1e-6)
    return iou

def evaluate():
    import glob
    model_paths = glob.glob(r"models\rtdetr_train*\weights\best.pt")
    if not model_paths:
        model_paths = glob.glob(r"runs\detect\models\rtdetr_train*\weights\best.pt")
    if not model_paths:
        print("Model not found. Run training first.")
        return
        
    best_model_path = model_paths[-1] # Latest run
    print(f"Evaluating model: {best_model_path}")
    model = RTDETR(best_model_path)
    
    test_dir = r"data\rtdetr_dataset\test\images"
    test_labels_dir = r"data\rtdetr_dataset\test\labels"
    
    test_images = glob.glob(os.path.join(test_dir, "*.jpg"))
    
    total_gt_points = 0
    total_det_points = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    exact_12_detected = 0
    exact_10_detected = 0
    other_counts = 0
    total_test_images = len(test_images)
    
    for img_path in test_images:
        basename = os.path.basename(img_path)
        lbl_path = os.path.join(test_labels_dir, basename.replace(".jpg", ".txt"))
        
        gt_points = []
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if int(parts[0]) == 1: # solder_point
                        gt_points.append([float(x) for x in parts[1:]])
                        
        total_gt_points += len(gt_points)
        
        results = model(img_path, verbose=False)[0]
        
        det_points = []
        boxes = results.boxes.xywhn.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        
        for box, cls, conf in zip(boxes, classes, confs):
            if int(cls) == 1 and conf > 0.25:
                det_points.append(box)
                
        num_det = len(det_points)
        total_det_points += num_det
        
        if num_det == 12: exact_12_detected += 1
        elif num_det == 10: exact_10_detected += 1
        else: other_counts += 1
        
        # Simple bipartite matching for TP/FP/FN using IoU > 0.3
        matched_gt = set()
        matched_det = set()
        for d_idx, d_box in enumerate(det_points):
            best_iou = 0
            best_g_idx = -1
            for g_idx, g_box in enumerate(gt_points):
                if g_idx in matched_gt: continue
                iou = compute_iou(d_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_g_idx = g_idx
            if best_iou > 0.3:
                matched_gt.add(best_g_idx)
                matched_det.add(d_idx)
                true_positives += 1
                
        false_positives += len(det_points) - len(matched_det)
        false_negatives += len(gt_points) - len(matched_gt)

    success_rate = (exact_12_detected / total_test_images) * 100 if total_test_images > 0 else 0
    
    with open(r"output\reports\evaluation_report.txt", "w") as f:
        f.write("=== RT-DETRv2-L TEST SET EVALUATION ===\n\n")
        f.write(f"Total Test Images: {total_test_images}\n")
        f.write(f"1. Total ground-truth solder points: {total_gt_points}\n")
        f.write(f"2. Total detected solder points: {total_det_points}\n")
        f.write(f"3. True positives: {true_positives}\n")
        f.write(f"4. False positives: {false_positives}\n")
        f.write(f"5. False negatives: {false_negatives}\n")
        f.write(f"7. Images with exactly 12 detected solder points: {exact_12_detected}\n")
        f.write(f"8. Images with exactly 10 detected solder points: {exact_10_detected}\n")
        f.write(f"9. Images with other counts: {other_counts}\n")
        f.write(f"10. Exact 12-point image success rate: {success_rate:.2f}%\n")
        
if __name__ == "__main__":
    evaluate()
