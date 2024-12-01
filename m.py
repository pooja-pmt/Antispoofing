from ultralytics import YOLO

model = YOLO("models/l_version_1_300.pt")
print(model.names)   # Check what classes it detects
