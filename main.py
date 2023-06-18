from networks.yolov8 import model

model = model.DetectionModel(cfg=f'yolov8n.yaml', ch=3, nc=None)

# model, ckpt = attempt_load_one_weight(weights) # NOT IMPLEMENTED

print(model)
