import os
import random
import shutil

INPUT_FOLDER = "Data/Labeled_augmented"

OUTPUT_BASE = "Data"
TRAIN_IMG = os.path.join(OUTPUT_BASE, "Train")
VAL_IMG = os.path.join(OUTPUT_BASE, "Test")
TRAIN_LBL = os.path.join(OUTPUT_BASE, "Train")
VAL_LBL = os.path.join(OUTPUT_BASE, "Test")

os.makedirs(TRAIN_IMG, exist_ok=True)
os.makedirs(VAL_IMG, exist_ok=True)
os.makedirs(TRAIN_LBL, exist_ok=True)
os.makedirs(VAL_LBL, exist_ok=True)

# Mapeo de clases
class_map = {
    15: 0, 16: 1, 17: 2, 18: 3, 19: 4,
    20: 5, 21: 6, 22: 7, 23: 8, 24: 9,
    25: 10, 26: 11, 27: 12
}

# Divisón en train y test

files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".jpg")]
random.shuffle(files)

split = int(0.8 * len(files))  # 80% train
train_files = files[:split]
val_files = files[split:]

# Función para copiar imágenes y corregir labels

def process_files(file_list, img_out, lbl_out):
    for file in file_list:
        img_path = os.path.join(INPUT_FOLDER, file)
        txt_path = img_path.replace(".jpg", ".txt")

        if not os.path.exists(txt_path):
            continue

        # Copiar imagen
        shutil.copy(img_path, os.path.join(img_out, file))

        # Reescribir labels
        with open(txt_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            cls = int(parts[0])

            if cls not in class_map:
                continue

            new_cls = class_map[cls]
            new_lines.append(f"{new_cls} {' '.join(parts[1:])}\n")

        # Guardar label
        new_txt = os.path.join(lbl_out, file.replace(".jpg", ".txt"))
        with open(new_txt, "w") as f:
            f.writelines(new_lines)

process_files(train_files, TRAIN_IMG, TRAIN_LBL)
process_files(val_files, VAL_IMG, VAL_LBL)

print("Dataset dividido y labels corregidos")