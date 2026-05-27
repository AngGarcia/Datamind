from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def main():
    model.train(
        data="Classes.yaml",
        epochs=100,
        imgsz=800,
        batch=8,
        name="cards_detector",
        device=0,
        workers=4
    )

if __name__ == "__main__":
    main()

# imgsz: tamaño al que se redimensionan las imágenes (800 o 960 para más detalle, 640 para más velocidad)
# batch: número de imágenes procesadas en cada iteración
# name="cards_detector": nombre del experimento, crea la carpeta runs/detect/cards_detector/ con resultados
# device=0: usa GPU si hay (="cpu" para usar solo CPU)
# workers=4: número de procesos para cargar datos