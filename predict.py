import os
import sys
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RT-DETR/rtdetrv2_pytorch'))
from src.core import YAMLConfig
import torch.nn as nn

def main():
    config_path = "RT-DETR/rtdetrv2_pytorch/configs/custom/rtdetrv2_baseline.yml"
    checkpoint_path = "models/rtdetrv2_train_baseline/best.pth"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu') 
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
        
    cfg.model.load_state_dict(state)
    
    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(device)
    model.eval()

    images_to_test = [
        "data/coco_dataset/val2017/PCBphoto-1-_jpg.rf.e2de8b7392e804d596b7ac9bd4ab5d3e.jpg",
        "data/coco_dataset/val2017/PCBphoto-120-_jpg.rf.60a328d08979635f324c67bb70a324e4.jpg",
        "data/coco_dataset/val2017/PCBphoto-126-_jpg.rf.3f3c48a92b1cbabd5b6b42db9845a7d4.jpg",
    ]
    
    class_names = {0: "group", 1: "solder_point"}

    transforms = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
    ])

    for img_path in images_to_test:
        im_pil = Image.open(img_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(device)
        
        im_data = transforms(im_pil)[None].to(device)
        
        with torch.no_grad():
            output = model(im_data, orig_size)
            
        labels, boxes, scores = output
        
        # draw
        draw = ImageDraw.Draw(im_pil)
        
        thrh = 0.3
        scr = scores[0]
        lab = labels[0][scr > thrh]
        box = boxes[0][scr > thrh]
        scrs = scores[0][scr > thrh]

        for j,b in enumerate(box):
            cls_id = lab[j].item()
            color = 'red' if cls_id == 0 else 'blue'
            draw.rectangle(list(b), outline=color, width=2)
            name = class_names.get(cls_id, str(cls_id))
            draw.text((b[0], max(0, b[1]-10)), text=f"{name} {scrs[j].item():.2f}", fill=color)

        out_name = os.path.basename(img_path).replace('.jpg', '_pred.jpg')
        im_pil.save(out_name)
        print(f"Saved {out_name}")

if __name__ == '__main__':
    main()
