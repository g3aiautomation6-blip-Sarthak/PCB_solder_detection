import os
import re
import csv
import argparse
import matplotlib.pyplot as plt

def parse_log(log_path, output_dir):
    epochs = []
    train_losses = []
    val_aps = []
    val_ap50s = []
    val_ars = []
    
    with open(log_path, 'r') as f:
        content = f.read()
        
    # Find all training stats.
    # Format: Averaged stats: lr: 0.000000  loss: 34.2794 (34.4121)
    loss_matches = re.findall(r"Averaged stats: lr:.*?loss: ([\d\.]+) \([\d\.]+\)", content)
    
    # Validation format: 
    # Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
    # Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.000
    # ...
    # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.005
    # (There are 12 metric lines per test loop)
    ap_matches = re.findall(r"Average Precision  \(AP\) @\[ IoU=0.50:0.95 \| area=   all \| maxDets=100 \] = ([-0-9\.]+)", content)
    ap50_matches = re.findall(r"Average Precision  \(AP\) @\[ IoU=0.50      \| area=   all \| maxDets=100 \] = ([-0-9\.]+)", content)
    ar_matches = re.findall(r"Average Recall     \(AR\) @\[ IoU=0.50:0.95 \| area=   all \| maxDets=100 \] = ([-0-9\.]+)", content)
    
    for i in range(len(loss_matches)):
        epochs.append(i)
        train_losses.append(float(loss_matches[i]))
        if i < len(ap_matches):
            val_aps.append(float(ap_matches[i]))
            val_ap50s.append(float(ap50_matches[i]))
            val_ars.append(float(ar_matches[i]))
        else:
            val_aps.append(0.0)
            val_ap50s.append(0.0)
            val_ars.append(0.0)
            
    csv_path = os.path.join(output_dir, 'training_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'TrainLoss', 'Val_AP', 'Val_AP50', 'Val_AR'])
        for i in range(len(epochs)):
            writer.writerow([epochs[i], train_losses[i], val_aps[i], val_ap50s[i], val_ars[i]])
            
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    
    plt.figure()
    plt.plot(epochs, val_ap50s, label='AP50', marker='x')
    plt.plot(epochs, val_aps, label='AP50:95', marker='x')
    plt.plot(epochs, val_ars, label='AR50:95 (maxDets=100)', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Validation Metrics')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'metrics_curve.png'))
    
    print("Parsed data:")
    for i in range(len(epochs)):
        print(f"Epoch {epochs[i]}: loss={train_losses[i]}, AP50={val_ap50s[i]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    parse_log(args.log, args.out)
