import os
import cv2
import matplotlib.pyplot as plt

FOLDER = "Prueba_augmentation"
SCALE = 0.6  # escala visual


def draw_yolo_boxes(image, txt_path):
    h, w = image.shape[:2]

    with open(txt_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            cls = parts[0]
            x, y, bw, bh = map(float, parts[1:])

            # YOLO → píxeles
            x1 = int((x - bw/2) * w)
            y1 = int((y - bh/2) * h)
            x2 = int((x + bw/2) * w)
            y2 = int((y + bh/2) * h)

            # Dibujar caja
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Texto clase
            cv2.putText(image, str(cls), (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return image


def show_all_images():
    files = [f for f in os.listdir(FOLDER) if f.endswith(".jpg")]

    for file in files:
        img_path = os.path.join(FOLDER, file)
        txt_path = img_path.replace(".jpg", ".txt")

        if not os.path.exists(txt_path):
            continue

        image = cv2.imread(img_path)
        image = draw_yolo_boxes(image, txt_path)

        # Escalar
        image = cv2.resize(image, None, fx=SCALE, fy=SCALE)

        # Convertir BGR → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Mostrar
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.title(file)
        plt.axis("off")
        plt.show()


def show_one_image(filename):
    img_path = os.path.join(FOLDER, filename)
    txt_path = img_path.replace(".jpg", ".txt")

    image = cv2.imread(img_path)
    image = draw_yolo_boxes(image, txt_path)

    image = cv2.resize(image, None, fx=SCALE, fy=SCALE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image)
    plt.title(filename)
    plt.axis("off")
    plt.show()


# ===== USO =====

# Ver una imagen:
# show_one_image("IMG20260519170724.jpg")

# Ver toda la carpeta:
show_all_images()