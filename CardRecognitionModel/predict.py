from ultralytics import YOLO
import cv2

# cargar modelo entrenado
model = YOLO("runs/detect/cards_detector/weights/best.pt")

# probar con imagen
results = model("IMG20260524105701.jpg")



# Imprimimos resultados como una lista

result = results[0]  # solo una imagen

boxes = result.boxes

classes = boxes.cls.tolist()      # Índices de clase

# Convertir a nombres
names = result.names
predicted_cards = [names[int(c)] for c in classes] # <- LA LISTA

print(predicted_cards)