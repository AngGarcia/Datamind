from ultralytics import YOLO

# cargar modelo entrenado
model = YOLO("runs/detect/cards_detector/weights/best.pt")

# probar con imagen
results = model("IMG_20221227_181822.jpg", show=True)
