import sys
import os

# Add official RT-DETR source to path
sys.path.insert(0, os.path.abspath(r"RT-DETR\rtdetrv2_pytorch"))

import torch
import yaml
from src.core import YAMLConfig

def sanity_check():
    print("=== RT-DETRv2-L SANITY CHECK ===")
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"CUDA Available: True ({torch.cuda.get_device_name(0)})")
    else:
        print("CUDA Available: False (Running sanity check on CPU)")
        
    config_dir = r"RT-DETR\rtdetrv2_pytorch\configs\rtdetrv2"
    config_path = os.path.join(config_dir, "rtdetrv2_hgnetv2_l_6x_coco.yml")
    
    with open(config_path, "r") as f:
        base_cfg = f.read()
        
    custom_cfg = base_cfg + "\nnum_classes: 2\n"
    custom_cfg_path = os.path.join(config_dir, "temp_sanity_cfg.yml")
    with open(custom_cfg_path, "w") as f:
        f.write(custom_cfg)
        
    old_cwd = os.getcwd()
    try:
        # Move to rtdetrv2_pytorch directory so yaml loading works
        os.chdir(r"RT-DETR\rtdetrv2_pytorch")
        cfg = YAMLConfig(r"configs\rtdetrv2\temp_sanity_cfg.yml", resume=None)
        
        # Load model
        model = cfg.model
        print(f"Model instantiated: {type(model).__name__}")
        print(f"Backbone: {type(model.backbone).__name__}")
        print(f"Classification head classes: {model.decoder.num_classes}")
        
        images = torch.randn(2, 3, 512, 512)
        
        targets = [
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
                "labels": torch.tensor([1], dtype=torch.int64)
            },
            {
                "boxes": torch.tensor([[0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]]),
                "labels": torch.tensor([0, 1], dtype=torch.int64)
            }
        ]
        
        print("Running forward pass...")
        outputs = model(images, targets)
        
        print("Calculating loss...")
        criterion = cfg.criterion
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        print("Running backward pass...")
        losses.backward()
        
        print("Optimizer step...")
        optimizer = cfg.optimizer
        optimizer.step()
        
        print("SANITY CHECK PASSED!")
        
    except Exception as e:
        print(f"SANITY CHECK FAILED: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        os.chdir(old_cwd)
        if os.path.exists(custom_cfg_path):
            os.remove(custom_cfg_path)

if __name__ == "__main__":
    sanity_check()
