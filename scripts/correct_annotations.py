import os
import glob

DATA_DIR = r"data\working_dataset"
OUT_LBL_DIR = r"data\corrected_labels"
REPORT_PATH = r"output\reports\correction_report.txt"

def correct_labels():
    os.makedirs(OUT_LBL_DIR, exist_ok=True)
    images = glob.glob(os.path.join(DATA_DIR, "*", "images", "*.jpg"))
    report_lines = []
    
    for img_path in images:
        basename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(basename)[0]
        split = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
        lbl_path = os.path.join(DATA_DIR, split, "labels", name_no_ext + ".txt")
        out_path = os.path.join(OUT_LBL_DIR, name_no_ext + ".txt")
        
        if not os.path.exists(lbl_path): continue
        
        with open(lbl_path, "r") as f:
            lines = f.readlines()
            
        pts = []
        grps = []
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            obj = {'box': [cx, cy, bw, bh], 'line': line.strip()}
            if cls_id == 0: grps.append(obj)
            elif cls_id == 1: pts.append(obj)
            
        # Assign points to groups
        grp_pts = {i: [] for i in range(len(grps))}
        for p in pts:
            px, py = p['box'][0], p['box'][1]
            best_g, best_d = -1, 9999
            for i, g in enumerate(grps):
                gx, gy = g['box'][0], g['box'][1]
                d = (px-gx)**2 + (py-gy)**2
                if d < best_d:
                    best_d, best_g = d, i
            if best_g != -1:
                grp_pts[best_g].append(p)
                
        new_pts = []
        for i, g in enumerate(grps):
            gp = grp_pts[i]
            if len(gp) < 3:
                # Estimate the missing points
                gx, gy, gw, gh = g['box']
                # The 3 points should be roughly at gx - 0.25*gw, gx, gx + 0.25*gw
                # Or we can just look at existing points and fill gaps
                expected_centers = [
                    (gx - 0.3*gw, gy),
                    (gx, gy),
                    (gx + 0.3*gw, gy)
                ]
                pw, ph = gw/3.5, gh/1.5 # approximate point size
                
                # Match existing points to expected centers
                matched = set()
                for p in gp:
                    px, py = p['box'][0], p['box'][1]
                    best_e, best_d = -1, 9999
                    for e_idx, e_c in enumerate(expected_centers):
                        d = (px-e_c[0])**2 + (py-e_c[1])**2
                        if d < best_d:
                            best_d, best_e = d, e_idx
                    if best_e != -1:
                        matched.add(best_e)
                
                # Add missing points
                for e_idx, e_c in enumerate(expected_centers):
                    if e_idx not in matched:
                        new_pts.append([e_c[0], e_c[1], pw, ph])
                        
        with open(out_path, "w") as f:
            for g in grps: f.write(g['line'] + "\n")
            for p in pts: f.write(p['line'] + "\n")
            for np_box in new_pts:
                f.write(f"1 {np_box[0]:.6f} {np_box[1]:.6f} {np_box[2]:.6f} {np_box[3]:.6f}\n")
                
        if len(new_pts) > 0:
            report_lines.append(f"{basename}: Added {len(new_pts)} points")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))

if __name__ == "__main__":
    correct_labels()
