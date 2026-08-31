import os
import sys
import glob
import json
import torch
import cv2

sys.path.insert(0, os.path.abspath(r"RT-DETR\rtdetrv2_pytorch"))
from src.core import YAMLConfig

def verify_dataset():
    print("\n--- 4. VERIFY DATASET ---")
    coco_train = r"data\coco_dataset\annotations\instances_train2017.json"
    coco_val = r"data\coco_dataset\annotations\instances_val2017.json"
    coco_test = r"data\coco_dataset\annotations\instances_test2017.json"
    
    counts = {12: 0, 11: 0, 10: 0, 9: 0}
    group_counts = {4: 0, 3: 0}
    splits = {'train': 0, 'val': 0, 'test': 0}
    
    for split_name, path in [('train', coco_train), ('val', coco_val), ('test', coco_test)]:
        with open(path) as f:
            data = json.load(f)
        splits[split_name] = len(data['images'])
        
        for img in data['images']:
            img_id = img['id']
            pts = sum(1 for a in data['annotations'] if a['image_id'] == img_id and a['category_id'] == 1)
            grps = sum(1 for a in data['annotations'] if a['image_id'] == img_id and a['category_id'] == 0)
            if pts in counts: counts[pts] += 1
            if grps in group_counts: group_counts[grps] += 1
            
    print(f"Splits -> Train: {splits['train']}, Val: {splits['val']}, Test: {splits['test']}")
    print(f"Point Distribution -> 12: {counts[12]}, 11: {counts[11]}, 10: {counts[10]}, 9: {counts[9]}")
    print(f"Group Distribution -> 4: {group_counts[4]}, 3: {group_counts[3]}")
    
    assert counts[12] == 84
    assert counts[11] == 21
    assert counts[10] == 14
    assert counts[9] == 3
    assert group_counts[4] == 119
    assert group_counts[3] == 3
    print("Dataset verification PASSED. Annotations match exact visible-only constraints.")

def main():
    print("=== FINAL PRE-TRAINING VERIFICATION ===")
    
    print("\n--- 1. VERIFY EXACT MODEL CONFIG ---")
    config_path = r"RT-DETR\rtdetrv2_pytorch\configs\custom\rtdetrv2_baseline.yml"
    
    old_cwd = os.getcwd()
    os.chdir(r"RT-DETR\rtdetrv2_pytorch")
    
    try:
        cfg = YAMLConfig(r"configs\custom\rtdetrv2_baseline.yml", resume=None)
        model = cfg.model
        
        print(f"Model Class: {type(model).__name__}")
        print(f"Backbone: {type(model.backbone).__name__}")
        print(f"Num Classes: {model.decoder.num_classes}")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        
        print("\n--- 2. VERIFY PRETRAINED WEIGHTS ---")
        cache_dir = os.path.expanduser(r"~/.cache/torch/hub/checkpoints")
        checkpoint_name = "PPHGNetV2_L_ssld_pretrained_from_paddle.pth"
        checkpoint_path = os.path.join(cache_dir, checkpoint_name)
        
        if os.path.exists(checkpoint_path):
            size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
            print(f"Checkpoint found locally: {checkpoint_path}")
            print(f"File size: {size_mb:.2f} MB")
            print("Internet access is NOT required during training for the backbone.")
        else:
            print(f"Checkpoint NOT found at {checkpoint_path}")
            print("It will be downloaded on first run. Please download it for offline use.")

        print("\n--- 3. VERIFY 2-CLASS HEAD ---")
        print(f"num_classes = {model.decoder.num_classes}")
        print("class 0 = group")
        print("class 1 = solder_point")
        print("Newly initialized layers (shape changed for 2 classes):")
        # In RT-DETRv2, the classification head is typically `decoder.class_embed`
        for name, param in model.named_parameters():
            if 'class_embed' in name or 'cls' in name:
                if param.shape[-1] == 2 or param.shape[0] == 2:
                    print(f"  - {name}: {param.shape}")
                    
        os.chdir(old_cwd)
        verify_dataset()
        os.chdir(r"RT-DETR\rtdetrv2_pytorch")
        
        print("\n--- 5. RUN ONE REAL GPU BATCH ---")
        if torch.cuda.is_available():
            print(f"CUDA Detected: True")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
            model = model.cuda()
            device = 'cuda'
        else:
            print("CUDA Detected: False (Running real batch on CPU for verification)")
            device = 'cpu'
            
        train_dataloader = cfg.train_dataloader
        print("Loading real batch from train_dataloader...")
        batch = next(iter(train_dataloader))
        
        # Batch is typically a tuple of (images, targets)
        images = batch[0].to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in batch[1]]
        
        print(f"Images shape: {images.shape}")
        print(f"Target example boxes: {targets[0]['boxes'].shape}")
        
        print("Running forward pass...")
        outputs = model(images, targets)
        
        print("Calculating loss...")
        criterion = cfg.criterion
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        print(f"Loss computed: {losses.item():.4f}")
        
        print("Running backward pass...")
        losses.backward()
        
        print("Optimizer step...")
        optimizer = cfg.optimizer
        optimizer.step()
        
        print("\nREAL-BATCH SANITY TEST PASSED!")
        print("Training environment is fully validated.")
        
    except Exception as e:
        print(f"VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        os.chdir(old_cwd)

if __name__ == "__main__":
    main()
