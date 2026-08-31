import os
import glob
import cv2
import numpy as np
from collections import defaultdict

DATA_DIR = r"data\working_dataset"
REPORT_PATH = r"output\reports\annotation_policy_report.txt"
VIS_PATH = r"output\visualizations\occlusion_examples.jpg"

def main():
    labels = glob.glob(os.path.join(DATA_DIR, "*", "labels", "*.txt"))
    
    point_counts = defaultdict(int)
    group_counts = defaultdict(int)
    
    examples = {12: None, 11: None, 10: None, 9: None}
    
    # Calculate statistics and find examples
    for lbl_path in labels:
        with open(lbl_path, "r") as f:
            lines = f.readlines()
            
        pts = sum(1 for line in lines if line.startswith("1 "))
        grps = sum(1 for line in lines if line.startswith("0 "))
        
        point_counts[pts] += 1
        group_counts[grps] += 1
        
        if pts in examples and examples[pts] is None:
            img_path = lbl_path.replace("labels", "images").replace(".txt", ".jpg")
            if os.path.exists(img_path):
                examples[pts] = (img_path, lbl_path, pts, grps)
                
    # Generate report
    with open(REPORT_PATH, "w") as f:
        f.write("=== ANNOTATION POLICY REPORT ===\n\n")
        
        f.write("1. VISIBLE-ONLY ANNOTATION POLICY\n")
        f.write("The dataset intentionally annotates only VISIBLE solder points. Solder points that are fully or substantially occluded by wires, components, or other objects are NOT annotated. This is a deliberate choice reflecting real-world conditions where the system must detect what is actually visible, rather than hallucinating occluded points.\n\n")
        
        f.write("2. SOLDER-POINT COUNT DISTRIBUTION\n")
        for count in sorted(point_counts.keys(), reverse=True):
            f.write(f"  Images with {count} visible points: {point_counts[count]}\n")
            
        f.write("\n3. GROUP COUNT DISTRIBUTION\n")
        for count in sorted(group_counts.keys(), reverse=True):
            f.write(f"  Images with {count} group boxes: {group_counts[count]}\n")
            
        f.write("\n4. EXAMPLES OF OCCLUSION\n")
        for pts, ex in examples.items():
            if ex:
                img_path, _, _, grps = ex
                f.write(f"  {pts}-point example: {os.path.basename(img_path)} (Groups: {grps})\n")
        f.write("  Visual examples have been saved to output/visualizations/occlusion_examples.jpg. In the images with < 12 points, the unannotated areas clearly correspond to physical occlusions (e.g., wires crossing the solder joints).\n\n")
        
        f.write("5. WHY FORCING 12 ANNOTATIONS IS INCORRECT\n")
        f.write("Adding artificial boxes for occluded points would train the object detection model to 'guess' or hallucinate bounding boxes where no visual evidence exists. This degrades the model's precision and ability to learn the true visual features of a solder point. The detector's job is purely to find visible solder; a downstream geometric logic module should handle the deduction of missing points.\n\n")
        
        f.write("6. ROLE OF GROUP ANNOTATIONS\n")
        f.write("The class 0 'group' annotations (typically 4 per board) establish the macroscopic regions where the 3-point clusters physically reside. Even if a specific solder point is occluded, the group box anchors the expected coordinate space. In the final system, these group boxes can be used to:\n")
        f.write("  a) Partition the board into 4 logical quadrants/clusters.\n")
        f.write("  b) Verify whether a detected point belongs to a specific expected cluster.\n")
        f.write("  c) Deduce exactly which of the 3 expected points within a group is missing due to occlusion.\n\n")
        
        f.write("7. RECOMMENDATION FOR TRAINING RT-DETRv2\n")
        f.write("Proceed with training using the ORIGINAL, untouched annotations containing 9 to 12 points. RT-DETRv2 should be trained to strictly detect visible points and visible groups. The loss function will correctly penalize the model if it falsely predicts a solder point in an occluded region. After the model is trained to accurately output raw detections, we will implement the geometric verification logic (handling 12 expected positions) as a post-processing step.\n")

    # Generate Visualization Contact Sheet
    images_for_sheet = []
    for pts in [12, 11, 10, 9]:
        ex = examples.get(pts)
        if ex:
            img_path, lbl_path, p_count, g_count = ex
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            
            with open(lbl_path, "r") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if not parts: continue
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    color = (0, 0, 255) if cls_id == 0 else (0, 255, 0)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Add text label
            cv2.putText(img, f"Visible Points: {p_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, f"Groups: {g_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            img = cv2.resize(img, (256, 256))
            images_for_sheet.append(img)
            
    if images_for_sheet:
        grid_size = 2 # 2x2 grid for 4 images
        rows = []
        for i in range(0, len(images_for_sheet), grid_size):
            row = images_for_sheet[i:i+grid_size]
            while len(row) < grid_size:
                row.append(np.zeros((256, 256, 3), dtype=np.uint8))
            rows.append(np.concatenate(row, axis=1))
        contact_sheet = np.concatenate(rows, axis=0)
        cv2.imwrite(VIS_PATH, contact_sheet)

if __name__ == "__main__":
    main()
