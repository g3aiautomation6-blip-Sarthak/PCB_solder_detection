import sys
import os
import argparse
import subprocess
import torch

def main():
    print("=== RT-DETRv2-L GPU TRAINING LAUNCHER ===")
    
    # 1. Automatic CUDA Detection and Validation
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. GPU is required for training.")
        print("Refusing to silently fall back to CPU for a long training run.")
        sys.exit(1)
        
    device_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024 ** 3)
    
    print(f"Detected {device_count} GPU(s).")
    print(f"GPU Model: {gpu_name}")
    print(f"VRAM: {vram_gb:.2f} GB")
    
    parser = argparse.ArgumentParser(description="Train RT-DETRv2-L on GPU")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (default: auto-detected based on VRAM)")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--config", type=str, default="configs/custom/rtdetrv2_baseline.yml", help="Path to training config")
    
    args = parser.parse_args()
    
    # Auto-recommend batch size if not specified
    if args.batch_size is None:
        if vram_gb < 8:
            args.batch_size = 2
            print("WARNING: VRAM < 8GB. Setting batch size to 2.")
        elif vram_gb < 16:
            args.batch_size = 4
        elif vram_gb < 24:
            args.batch_size = 8
        else:
            args.batch_size = 16
            
    print(f"Configured Batch Size: {args.batch_size}")
    print(f"Configured Epochs: {args.epochs}")
    
    # In official RT-DETR, configs are modified by overriding from CLI or rewriting the YAML.
    # To keep this launcher robust, we'll execute the torchrun command.
    
    # Ensure working directory is the RT-DETR repo
    repo_dir = r"RT-DETR\rtdetrv2_pytorch"
    if not os.path.exists(repo_dir):
        print("ERROR: Official repository not found at", repo_dir)
        sys.exit(1)
        
    cmd = [
        "torchrun", "--nproc_per_node=1", "tools/train.py",
        "-c", os.path.abspath(args.config)
    ]
    
    if args.resume:
        cmd.extend(["--resume", args.resume])
    else:
        # Transfer learning from the EMA checkpoint if not resuming a mid-training run
        cmd.extend(["--tuning", "rtdetrv2_r50vd_120e_coco_ema.pth"])
        
    print(f"\nExecuting command:\n{' '.join(cmd)}\n")
    
    # Start the training process
    os.chdir(repo_dir)
    subprocess.run(cmd, check=True)
    
if __name__ == "__main__":
    main()
