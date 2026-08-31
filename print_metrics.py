import json
for line in open('models/rtdetrv2_train_solder_only/log.txt'):
    d = json.loads(line)
    print(f"Epoch {d['epoch']}: AP50={d['test_coco_eval_bbox'][1]:.3f}, AP50:95={d['test_coco_eval_bbox'][0]:.3f}, AR={d['test_coco_eval_bbox'][8]:.3f}")
