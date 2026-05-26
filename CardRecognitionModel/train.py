from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="Classes.yaml",
    epochs=100,
    imgsz=800,       # YOLO resizea las imágenes a 640x640 (para fijarte en detallitos, prueba 800 o 960)
    batch=8,        # procesa 16 imágenes por iteración (para CPU prueba 4-8, para GPU media, usa 16)
    name="cards_detector", # crea la carpeta runs/detect/cards_detector/ con resultados
    #device=0,        # usa GPU si hay (="cpu" si lo prefieres)
    workers=4        # acelera carga de datos
)



# pip install ultralytics

# ¿Qué hace?
# descarga pesos preentrenados
# entrena tu modelo
# guarda resultados en:
# runs/detect/cards_detector/


# Meaning of the training:

#    Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
# 1/100         0G      1.322      4.406      1.469          8        800: 0% ──────────── 1/1751 62.1s/it 38.7s<30:
# 1/100         0G      1.334      4.417       1.47          8        800: 0% ──────────── 2/1751 36.3s/it 57.1s<17:

# The epoch we're in (1 of 100)
# GPU_mem is 0G because we're using CPU (if you use GPU it will show the memory used)
# box_loss: error in bounding box position/size (1.3 is normal in early training)
# cls_loss: error in classification (4.4 is high, but it's still early)
# dfl_loss: error in bounding box distribution focal loss
# Instances: number of objects detected in this batch (8 is good, matches batch size)
# Size: images are resized to 800x800 for training (YOLO default is 640, but 800 can help with small details)
# 1/1751 62.1s/it means we have 1751 batches per epoch and each iteration takes about 62.1 seconds