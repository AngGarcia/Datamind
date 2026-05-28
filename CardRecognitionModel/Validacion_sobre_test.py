
from ultralytics import YOLO

model = YOLO("runs/detect/cards_detector/weights/best.pt")

model.val(data="Classes.yaml", split="test")