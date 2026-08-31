import os
import sys
import json
import torch
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../RT-DETR/rtdetrv2_pytorch'))
from src.core import YAMLConfig
import torch.nn as nn
import torchvision.transforms as T
from pycocotools.coco import COCO

def calculate_iou(box1, box2):
    # box: [x, y, w, h]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0]+box1[2], box2[0]+box2[2])
    y2 = min(box1[1]+box1[3], box2[1]+box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area
    if union_area == 0: return 0
    return inter_area / union_area

def main():
    config_path = "RT-DETR/rtdetrv2_pytorch/configs/custom/rtdetrv2_solder_only_100epoch.yml"
    checkpoint_path = "models/baseline_B_100epoch/best.pth"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    val_ann_file = "data/coco_dataset_solder_only/annotations/instances_val2017.json"
    val_img_dir = "data/coco_dataset/val2017/"
    
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
            return self.postprocessor(self.model(images), orig_target_sizes)

    model = Model().to(device)
    model.eval()
    
    coco_gt = COCO(val_ann_file)
    transforms = T.Compose([T.Resize((512, 512)), T.ToTensor()])
    
    results = []
    
    print("Generating predictions...")
    for img_id in coco_gt.getImgIds():
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(val_img_dir, img_info['file_name'])
        
        im_pil = Image.open(img_path).convert('RGB')
        orig_size = torch.tensor([im_pil.size[0], im_pil.size[1]])[None].to(device)
        im_data = transforms(im_pil)[None].to(device)
        
        with torch.no_grad():
            labels, boxes, scores = model(im_data, orig_size)
            
        for i in range(len(scores[0])):
            box = boxes[0][i].tolist()
            w, h = box[2] - box[0], box[3] - box[1]
            results.append({
                "image_id": img_id,
                "bbox": [box[0], box[1], w, h],
                "score": scores[0][i].item()
            })
            
    thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    print(f"{'Threshold':<10}{'TP':<5}{'FP':<5}{'FN':<5}{'Precision':<12}{'Recall':<10}{'F1':<10}{'Mean Count':<15}{'Exact Acc':<15}{'Under-rate':<15}{'Over-rate':<15}")
    
    for t in thresholds:
        tp, fp, fn = 0, 0, 0
        exact, under, over = 0, 0, 0
        total_pred = 0
        
        for img_id in coco_gt.getImgIds():
            gts = [ann['bbox'] for ann in coco_gt.imgToAnns[img_id]]
            preds = [r for r in results if r['image_id'] == img_id and r['score'] >= t]
            preds.sort(key=lambda x: x['score'], reverse=True)
            
            total_pred += len(preds)
            if len(preds) == len(gts): exact += 1
            elif len(preds) < len(gts): under += 1
            else: over += 1
            
            matched_gt = set()
            for p in preds:
                best_iou = 0
                best_gt = -1
                for i, gt in enumerate(gts):
                    if i in matched_gt: continue
                    iou = calculate_iou(p['bbox'], gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = i
                if best_iou >= 0.5:
                    matched_gt.add(best_gt)
                    tp += 1
                else:
                    fp += 1
            fn += len(gts) - len(matched_gt)
            
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        mean_cnt = total_pred / len(coco_gt.getImgIds())
        exact_acc = exact / len(coco_gt.getImgIds())
        under_r = under / len(coco_gt.getImgIds())
        over_r = over / len(coco_gt.getImgIds())
        
        print(f"{t:<10.2f}{tp:<5}{fp:<5}{fn:<5}{prec:<12.4f}{rec:<10.4f}{f1:<10.4f}{mean_cnt:<15.2f}{exact_acc:<15.4f}{under_r:<15.4f}{over_r:<15.4f}")

if __name__ == '__main__':
    main()
