import os
import sys
import torch
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../RT-DETR/rtdetrv2_pytorch'))
from src.core import YAMLConfig
import torch.nn as nn
import torchvision.transforms as T
from pycocotools.coco import COCO

def calculate_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[0]+box1[2], box2[0]+box2[2]), min(box1[1]+box1[3], box2[1]+box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = box1[2]*box1[3] + box2[2]*box2[3] - inter
    return inter / union if union > 0 else 0

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
            self.model, self.postprocessor = cfg.model.deploy(), cfg.postprocessor.deploy()
        def forward(self, im, orig): return self.postprocessor(self.model(im), orig)

    model = Model().to(device)
    model.eval()
    
    coco = COCO(val_ann_file)
    transforms = T.Compose([T.Resize((512, 512)), T.ToTensor()])
    
    best_threshold = 0.5 # Default, should change if F1 max is elsewhere
    out_dir = "models/baseline_B_100epoch/visualizations"
    os.makedirs(out_dir, exist_ok=True)
    
    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(val_img_dir, img_info['file_name'])
        
        im_pil = Image.open(img_path).convert('RGB')
        orig_size = torch.tensor([im_pil.size[0], im_pil.size[1]])[None].to(device)
        im_data = transforms(im_pil)[None].to(device)
        
        with torch.no_grad():
            labels, boxes, scores = model(im_data, orig_size)
            
        gts = [ann['bbox'] for ann in coco.imgToAnns[img_id]]
        preds = []
        for i in range(len(scores[0])):
            if scores[0][i] >= best_threshold:
                box = boxes[0][i].tolist()
                preds.append({"bbox": [box[0], box[1], box[2]-box[0], box[3]-box[1]], "score": scores[0][i].item()})
                
        # Draw GT in Green, Pred in Red
        draw = ImageDraw.Draw(im_pil)
        for gt in gts:
            draw.rectangle([gt[0], gt[1], gt[0]+gt[2], gt[1]+gt[3]], outline="green", width=2)
        for p in preds:
            b = p['bbox']
            draw.rectangle([b[0], b[1], b[0]+b[2], b[1]+b[3]], outline="red", width=2)
            draw.text((b[0], b[1]-10), text=f"{p['score']:.2f}", fill="red")
            
        out_name = os.path.join(out_dir, f"gt{len(gts)}_pred{len(preds)}_{img_info['file_name']}")
        im_pil.save(out_name)

if __name__ == '__main__':
    main()
