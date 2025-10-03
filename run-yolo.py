from ultralytics import YOLO

model = YOLO(model="yolo11n-cls.pt")

print(model.model[0])