import os
import sys
import json
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RT-DETR/rtdetrv2_pytorch'))
from src.core import YAMLConfig
import torch.nn as nn

def box_iou(box1, box2):
    # box: [cx, cy, w, h] to [x1, y1, x2, y2]
    b1_x1, b1_y1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    b1_x2, b1_y2 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    b2_x1, b2_y1 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    b2_x2, b2_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
    
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0
        
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    iou = inter_area / (b1_area + b2_area - inter_area)
    return iou

def evaluate():
    config_path = "RT-DETR/rtdetrv2_pytorch/configs/custom/rtdetrv2_solder_only.yml"
    checkpoint_path = "models/rtdetrv2_train_solder_only/best.pth"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu') 
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    cfg.model.load_state_dict(state)
    
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(device)
    model.eval()

    transforms = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
    ])

    with open("data/coco_dataset_solder_only/annotations/instances_val2017.json") as f:
        coco = json.load(f)
    
    img_dict = {img['id']: img for img in coco['images']}
    ann_dict = {img['id']: [] for img in coco['images']}
    for ann in coco['annotations']:
        ann_dict[ann['image_id']].append(ann)
        
    thrh = 0.6
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    exact_matches = 0
    under_counts = 0
    over_counts = 0
    
    images_to_viz = {} # key: expected_count
    
    for img_id, img_info in img_dict.items():
        img_path = os.path.join("data/coco_dataset/val2017", img_info['file_name'])
        im_pil = Image.open(img_path).convert('RGB')
        w, h = im_pil.size
        
        orig_size = torch.tensor([w, h])[None].to(device)
        im_data = transforms(im_pil)[None].to(device)
        
        with torch.no_grad():
            output = model(im_data, orig_size)
            
        labels, boxes, scores = output
        
        scr = scores[0]
        mask = scr > thrh
        pred_boxes = boxes[0][mask].cpu().numpy()
        pred_scores = scr[mask].cpu().numpy()
        
        # COCO format is [x, y, w, h] (top-left) -> convert to [cx, cy, w, h] to match our iou func, wait, 
        # Actually our box_iou expects [cx, cy, w, h]. Wait, RT-DETR prediction is [x1, y1, x2, y2]
        # Let's adjust box_iou or convert. RT-DETR outputs [x1, y1, x2, y2].
        
        def iou_xyxy(b1, b2):
            inter_x1 = max(b1[0], b2[0])
            inter_y1 = max(b1[1], b2[1])
            inter_x2 = min(b1[2], b2[2])
            inter_y2 = min(b1[3], b2[3])
            if inter_x2 < inter_x1 or inter_y2 < inter_y1: return 0.0
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            b1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
            b2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])
            return inter_area / (b1_area + b2_area - inter_area)
            
        gt_boxes = []
        for ann in ann_dict[img_id]:
            x, y, bw, bh = ann['bbox']
            gt_boxes.append([x, y, x+bw, y+bh])
            
        gt_boxes = np.array(gt_boxes)
        pred_count = len(pred_boxes)
        gt_count = len(gt_boxes)
        
        # Match
        matched_gt = set()
        tp = 0
        fp = 0
        
        # Sort predictions by score descending
        sort_idx = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[sort_idx]
        pred_scores = pred_scores[sort_idx]
        
        for pb in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            for j, gb in enumerate(gt_boxes):
                if j in matched_gt: continue
                iou = iou_xyxy(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            if best_iou > 0.5:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
                
        fn = gt_count - tp
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        if pred_count == gt_count:
            exact_matches += 1
        elif pred_count < gt_count:
            under_counts += 1
        else:
            over_counts += 1
            
        # Draw for visualization
        if gt_count not in images_to_viz:
            draw = ImageDraw.Draw(im_pil)
            for pb, score in zip(pred_boxes, pred_scores):
                draw.rectangle(list(pb), outline='blue', width=2)
                draw.text((pb[0], max(0, pb[1]-10)), text=f"solder {score:.2f}", fill='blue')
            for gb in gt_boxes:
                draw.rectangle(list(gb), outline='green', width=1) # GT in green
            out_name = f"viz_{gt_count}_points.jpg"
            im_pil.save(out_name)
            images_to_viz[gt_count] = out_name

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    total_imgs = len(img_dict)
    
    print(f"Total Images: {total_imgs}")
    print(f"Exact Visible Count Accuracy: {exact_matches / total_imgs * 100:.2f}% ({exact_matches}/{total_imgs})")
    print(f"Under-count Rate: {under_counts / total_imgs * 100:.2f}%")
    print(f"Over-count Rate: {over_counts / total_imgs * 100:.2f}%")
    print(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    
    for count, path in images_to_viz.items():
        print(f"Saved visualization for {count} points: {path}")

if __name__ == '__main__':
    evaluate()
