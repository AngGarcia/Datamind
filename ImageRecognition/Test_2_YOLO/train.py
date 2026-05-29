from ultralytics import YOLO

# cargar modelo base (ligero)
model = YOLO("yolov8n.pt")

# entrenar
model.train(
    data="cards.yaml",
    epochs=10,
    imgsz=640,
    batch=16,
    name="cards_detector"
)
# subir las epochs a 50 por lo menos...

# pip install ultralytics
# python train.py


# ¿Qué hace?
# descarga pesos preentrenados
# entrena tu modelo
# guarda resultados en:
# runs/detect/cards_detector/