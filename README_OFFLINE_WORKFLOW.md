# Offline Deployment Workflow

This PCB Solder Detection experiment utilizes a dual-machine workflow because the primary development PC lacks a CUDA-capable NVIDIA GPU.

## Step 1: GPU Training Machine Setup
1. Copy this entire project directory (`PCB_Solder_detection`) to a machine equipped with a modern NVIDIA GPU (e.g., RTX 3090, 4090, or cloud instance).
2. Install dependencies:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install tensorboard scipy faster_coco_eval PyYAML
   ```
3. Run the GPU training script:
   ```bash
   python scripts/train_rtdetrv2_gpu.py --config RT-DETR/rtdetrv2_pytorch/configs/custom/rtdetrv2_baseline.yml --epochs 100
   ```
   *(Note: The script will automatically scale batch size based on available VRAM and fail safely if CUDA is not detected).*

## Step 2: Extract Trained Checkpoint
Once training is complete, the `lyuwenyu/RT-DETR` pipeline will output the trained weights to:
`models/rtdetrv2_train_baseline/best.pth`

## Step 3: Copy to Offline PC
1. Copy the `best.pth` checkpoint file back to the primary offline Windows PC (the development machine).
2. Place it in `models/rtdetrv2_train_baseline/best.pth`.

## Step 4: Local Evaluation & Future Phase 3
1. With the trained weights available locally, you can evaluate the raw detector performance on the TEST split using:
   ```bash
   python scripts/evaluate_rtdetrv2_gpu.py
   ```
2. The final trained model will then be integrated into the Phase 3 GUI and combined with the 12-position geometry logic for strict layout validation.
