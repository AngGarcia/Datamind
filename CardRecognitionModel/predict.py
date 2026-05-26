from ultralytics import YOLO
import cv2

# cargar modelo entrenado
model = YOLO("runs/detect/cards_detector/weights/best.pt")

# probar con imagen
results = model("IMG20260524105701.jpg") #, show=True)




# Obtener imagen con anotaciones
annotated_img = results[0].plot()

# reducir tamaño (por ejemplo al 50%, porque si no se genera enorme y no se puede reducir el tamaño)
scale = 0.3
resized = cv2.resize(annotated_img, None, fx=scale, fy=scale)

# mostrar
cv2.imshow("Resultado", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# guardar imagen
#cv2.imwrite("resultado.jpg", resized)