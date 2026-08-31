import os
from ultralytics import RTDETR

def train_rtdetr():
    # Model candidate
    model_name = "rtdetr-l.pt"
    
    # Load a pretrained RT-DETR model
    model = RTDETR(model_name)
    
    data_yaml = os.path.abspath(r"data\rtdetr_dataset\dataset.yaml")
    
    print("Starting training of RT-DETRv2-L (transfer learning)...")
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=3,          # Small number of epochs for the experimental phase
        imgsz=512,         # Same as our dataset
        batch=4,           # Conservative batch size
        project='models',
        name='rtdetr_train',
        device='0' if os.environ.get("CUDA_VISIBLE_DEVICES") else 'cpu', # Use CPU if no GPU
        pretrained=True,
        # Realistic augmentation
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        perspective=0.0,
        fliplr=0.5,
        seed=42,
        val=True
    )
    
    print("Training complete. Results saved to models/rtdetr_train/")

if __name__ == "__main__":
    train_rtdetr()
