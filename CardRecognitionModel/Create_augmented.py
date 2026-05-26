import os
import cv2
import albumentations as A
import shutil

INPUT_FOLDER = "Data/Labeled"
OUTPUT_FOLDER = "Data/Labeled_augmented"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Transformaciones
transforms = [
    ("rot_right_small", A.Rotate(limit=(3, 3), p=1)),
    ("rot_right_big", A.Rotate(limit=(7, 7), p=1)),
    ("rot_left_small", A.Rotate(limit=(-3, -3), p=1)),
    ("rot_left_big", A.Rotate(limit=(-7, -7), p=1)),
    ("rot_180", A.Rotate(limit=(180, 180), p=1)),
    ("zoom_in", A.RandomScale(scale_limit=(0.2, 0.2), p=1)),
    ("zoom_out", A.RandomScale(scale_limit=(-0.2, -0.2), p=1)),
    ("bright", A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0, p=1)),
    ("dark", A.RandomBrightnessContrast(brightness_limit=-0.2, contrast_limit=0, p=1)),
]


def load_yolo_labels(txt_path):
    bboxes = []
    class_labels = []
    with open(txt_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            class_labels.append(int(parts[0]))
            bbox = list(map(float, parts[1:]))
            bboxes.append(bbox)
    return bboxes, class_labels


def save_yolo_labels(txt_path, bboxes, class_labels):
    with open(txt_path, "w") as f:
        for box, label in zip(bboxes, class_labels):
            label = int(label)  # PONER INT porque 
            f.write(f"{label} {' '.join(map(str, box))}\n")


# Loop principal (solo uno)
for file in os.listdir(INPUT_FOLDER):
    if file.endswith(".jpg") and "_" not in file:

        img_path = os.path.join(INPUT_FOLDER, file)
        txt_path = img_path.replace(".jpg", ".txt")

        if not os.path.exists(txt_path):
            continue

        base_name = file.replace(".jpg", "")

        # 1. Copiar original
        shutil.copy(img_path, os.path.join(OUTPUT_FOLDER, file))
        shutil.copy(txt_path, os.path.join(OUTPUT_FOLDER, base_name + ".txt"))

        image = cv2.imread(img_path)
        bboxes, class_labels = load_yolo_labels(txt_path)

        # 2. Generar augmentaciones
        for name, transform in transforms:

            aug = A.Compose(
                [transform],
                bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
            )

            augmented = aug(image=image, bboxes=bboxes, class_labels=class_labels)

            new_img = augmented["image"]
            new_bboxes = augmented["bboxes"]
            new_labels = augmented["class_labels"]

            new_img_name = f"{base_name}_{name}.jpg"
            new_txt_name = f"{base_name}_{name}.txt"

            cv2.imwrite(os.path.join(OUTPUT_FOLDER, new_img_name), new_img)

            save_yolo_labels(
                os.path.join(OUTPUT_FOLDER, new_txt_name),
                new_bboxes,
                new_labels
            )

print("✅ Dataset aumentado + originales copiados")