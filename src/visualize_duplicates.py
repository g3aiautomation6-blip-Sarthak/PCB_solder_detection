import cv2
import numpy as np

images = [
    r"data\working_dataset\train\images\PCBphoto-10-_jpg.rf.25fdb68be0b0c1dc409d38c3f1da8fff.jpg",
    r"data\working_dataset\train\images\PCBphoto-11-_jpg.rf.bb6a19161a18dd91eb35ed910044d49b.jpg",
    r"data\working_dataset\train\images\PCBphoto-5-_jpg.rf.1bce6c2e27a40aef535e7f01bfaca322.jpg"
]

loaded = [cv2.imread(p) for p in images]
vis = np.concatenate(loaded, axis=1)
cv2.imwrite(r"output\visualizations\duplicate_sample.jpg", vis)
