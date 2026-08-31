import glob

labels = glob.glob(r"data\working_dataset\*\labels\*.txt")

found_12 = False
found_10 = False

for l in labels:
    with open(l) as f:
        lines = f.readlines()
        pts = [line for line in lines if line.startswith("1 ")]
        grps = [line for line in lines if line.startswith("0 ")]
        
        if len(pts) == 12 and not found_12:
            print("=== 12-point ===")
            print(l)
            for g in grps: print(g.strip())
            for p in pts: print(p.strip())
            found_12 = True
            
        if len(pts) == 10 and not found_10:
            print("\n=== 10-point ===")
            print(l)
            for g in grps: print(g.strip())
            for p in pts: print(p.strip())
            found_10 = True
            
    if found_12 and found_10:
        break
