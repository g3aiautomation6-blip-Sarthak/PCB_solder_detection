import glob, os

labels = glob.glob(r"data\corrected_labels\*.txt")
failures = 0
for l in labels:
    with open(l) as f:
        lines = f.readlines()
        pts = sum(1 for x in lines if x.startswith("1 "))
        grps = sum(1 for x in lines if x.startswith("0 "))
        if pts != 12 or grps != 4:
            print(f"{os.path.basename(l)}: pts={pts}, grps={grps}")
            failures += 1

if failures == 0:
    print(f"All {len(labels)} images verified: exactly 12 points, 4 groups.")
