import json, os

COCO_DIR = r"data\coco_dataset"
with open(os.path.join(COCO_DIR, "annotations", "instances_train2017.json")) as f:
    train_data = json.load(f)
with open(os.path.join(COCO_DIR, "annotations", "instances_test2017.json")) as f:
    test_data = json.load(f)

img_to_move = train_data["images"].pop()
test_data["images"].append(img_to_move)

anns_to_move = [a for a in train_data["annotations"] if a["image_id"] == img_to_move["id"]]
train_data["annotations"] = [a for a in train_data["annotations"] if a["image_id"] != img_to_move["id"]]
test_data["annotations"].extend(anns_to_move)

with open(os.path.join(COCO_DIR, "annotations", "instances_train2017.json"), "w") as f:
    json.dump(train_data, f)
with open(os.path.join(COCO_DIR, "annotations", "instances_test2017.json"), "w") as f:
    json.dump(test_data, f)

import shutil
src = os.path.join(COCO_DIR, "train2017", img_to_move["file_name"])
dst = os.path.join(COCO_DIR, "test2017", img_to_move["file_name"])
if os.path.exists(src):
    shutil.move(src, dst)
