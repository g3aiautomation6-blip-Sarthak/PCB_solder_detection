import os
import subprocess

def run_rtdetrv2_baseline():
    print("=== RT-DETRv2-L BASELINE TRAINING SCRIPT ===")
    print("This script is intended to be run on an NVIDIA GPU.")
    print("Due to hardware limitations (CPU-only), this execution is a pipeline verify only.")
    
    # Official RT-DETR command format:
    # torchrun --nproc_per_node=1 tools/train.py -c configs/rtdetrv2/rtdetrv2_r50vd_120e_coco.yml
    
    config_content = """
__include__: [
  '../dataset/coco_detection.yml',
  '../runtime.yml',
  './include/dataloader.yml',
  './include/optimizer.yml',
  './include/rtdetrv2_r50vd.yml',
]

task: detection

output_dir: ../../../models/rtdetrv2_train_baseline

# Custom Dataset modifications
train_dataloader:
  dataset:
    img_folder: ../../../data/coco_dataset/train2017/
    ann_file: ../../../data/coco_dataset/annotations/instances_train2017.json
  batch_size: 4 # Reduced for initial baseline

val_dataloader:
  dataset:
    img_folder: ../../../data/coco_dataset/val2017/
    ann_file: ../../../data/coco_dataset/annotations/instances_val2017.json

num_classes: 2
epoches: 100 # Sensible baseline for transfer learning
"""
    os.makedirs(r"RT-DETR\rtdetrv2_pytorch\configs\custom", exist_ok=True)
    with open(r"RT-DETR\rtdetrv2_pytorch\configs\custom\rtdetrv2_baseline.yml", "w") as f:
        f.write(config_content)
        
    print("Configuration created at RT-DETR/rtdetrv2_pytorch/configs/custom/rtdetrv2_baseline.yml")
    print("To run training with GPU, execute:")
    print("cd RT-DETR/rtdetrv2_pytorch")
    print("pip install -r requirements.txt")
    print("torchrun --nproc_per_node=1 tools/train.py -c configs/custom/rtdetrv2_baseline.yml --tuning path/to/rtdetrv2_r50vd_120e_coco_ema.pth")

if __name__ == "__main__":
    run_rtdetrv2_baseline()
