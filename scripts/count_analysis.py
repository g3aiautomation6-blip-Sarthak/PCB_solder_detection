import os, sys, torch, json
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../RT-DETR/rtdetrv2_pytorch'))
from src.core import YAMLConfig
import torch.nn as nn
import torchvision.transforms as T
from pycocotools.coco import COCO

cfg = YAMLConfig('RT-DETR/rtdetrv2_pytorch/configs/custom/rtdetrv2_solder_only_100epoch.yml', resume='models/baseline_B_100epoch/best.pth')
checkpoint = torch.load('models/baseline_B_100epoch/best.pth', map_location='cpu')
state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
cfg.model.load_state_dict(state)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
    def forward(self, im, orig): return self.postprocessor(self.model(im), orig)

model = Model().to('cuda:0')
model.eval()

coco = COCO('data/coco_dataset_solder_only/annotations/instances_val2017.json')
transforms = T.Compose([T.Resize((512, 512)), T.ToTensor()])

results = []
for img_id in coco.getImgIds():
    img_info = coco.loadImgs(img_id)[0]
    im_pil = Image.open(os.path.join('data/coco_dataset/val2017/', img_info['file_name'])).convert('RGB')
    orig_size = torch.tensor([im_pil.size[0], im_pil.size[1]])[None].to('cuda:0')
    with torch.no_grad():
        labels, boxes, scores = model(transforms(im_pil)[None].to('cuda:0'), orig_size)
    gts = len(coco.imgToAnns[img_id])
    preds = sum(1 for s in scores[0] if s >= 0.65)
    results.append({'file': img_info['file_name'], 'gt': gts, 'pred': preds})

print('COUNT ANALYSIS')
print('filename | GT | Predicted | Diff')
for r in results:
    print(r['file'] + ' | ' + str(r['gt']) + ' | ' + str(r['pred']) + ' | ' + str(r['pred'] - r['gt']))

print('CATEGORIZED:')
for gt_val in [12, 11, 10, 9]:
    subset = [r for r in results if r['gt'] == gt_val]
    for r in subset:
        print('GT ' + str(gt_val) + ' -> predicted ' + str(r['pred']) + ' (' + r['file'] + ')')
