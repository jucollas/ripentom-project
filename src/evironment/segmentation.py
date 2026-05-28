import os
import hashlib
import shutil
import zipfile
import random
import cv2
import numpy as np
from PIL import Image, ImageEnhance

TARGET_PER_CLASS = 400
CLASES = ["ripe", "unripe", "damaged"]
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
OUTPUT_DIR = "data/raw/"

def segmentar_tomate_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    mask_sat = cv2.inRange(S, 60, 255)
    mask_val = cv2.inRange(V, 50, 255)

    lower_red1   = np.array([0,   70,  50]);  upper_red1   = np.array([10,  255, 255])
    lower_red2   = np.array([170, 70,  50]);  upper_red2   = np.array([180, 255, 255])
    lower_orange = np.array([10,  70,  50]);  upper_orange = np.array([25,  255, 255])
    lower_yellow = np.array([25,  70,  50]);  upper_yellow = np.array([35,  255, 255])
    lower_green  = np.array([35,  50,  50]);  upper_green  = np.array([85,  255, 255])

    mask_color = (
        cv2.inRange(hsv, lower_red1,   upper_red1)   |
        cv2.inRange(hsv, lower_red2,   upper_red2)   |
        cv2.inRange(hsv, lower_orange, upper_orange) |
        cv2.inRange(hsv, lower_yellow, upper_yellow) |
        cv2.inRange(hsv, lower_green,  upper_green)
    )

    mask = mask_sat & mask_val & mask_color

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_final = np.zeros_like(mask)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 500:
            cv2.drawContours(mask_final, [largest], -1, 255, -1)

    resultado = cv2.bitwise_and(img, img, mask=mask_final)
    return resultado, mask_final

def copiar_y_segmentar(src, dst):
    """Lee la imagen, la segmenta y la guarda en dst."""
    img = cv2.imread(src)
    if img is None:
        shutil.copy2(src, dst)  # fallback si no se puede leer
        return
    segmentada, _ = segmentar_tomate_hsv(img)
    cv2.imwrite(dst, segmentada)

# ─────────────────────────────────────────────
# 3. HASHING Y DETECCIÓN DE ÚNICOS
# ─────────────────────────────────────────────
def get_hash(filepath):
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def obtener_unicas_por_clase(dataset_path):
    unicas = {c: [] for c in CLASES}
    for clase in CLASES:
        hashes_vistos = {}
        for root, _, files in os.walk(dataset_path):
            for fname in files:
                if os.path.splitext(fname)[1].lower() not in EXTENSIONES:
                    continue
                if clase.lower() not in fname.lower():
                    continue
                fpath = os.path.join(root, fname)
                h = get_hash(fpath)
                if h not in hashes_vistos:
                    hashes_vistos[h] = fpath
        unicas[clase] = list(hashes_vistos.values())
        print(f"  [{clase}] imagenes unicas encontradas: {len(unicas[clase])}")
    return unicas

# ─────────────────────────────────────────────
# 4. DATA AUGMENTATION (3 métodos balanceados)
# ─────────────────────────────────────────────
def aug_flip(img_pil):
    return img_pil.transpose(Image.FLIP_LEFT_RIGHT)

def aug_rotate(img_pil):
    angle = random.uniform(-25, 25)
    return img_pil.rotate(angle, resample=Image.BILINEAR, expand=False)

def aug_brightness(img_pil):
    factor = random.uniform(0.7, 1.4)
    return ImageEnhance.Brightness(img_pil).enhance(factor)

AUGMENTATIONS = [aug_flip,   aug_rotate,   aug_brightness]
AUG_NAMES     = ["flip",     "rotate",     "brightness"]

def pil_to_cv2(img_pil):
    return cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

def aplicar_augmentation(imagenes_src, clase_dir, cantidad_necesaria):
    """
    Genera imagenes aumentadas, las segmenta y las guarda con nombre unripe_XXXX.jpg
    """
    por_metodo = cantidad_necesaria // len(AUGMENTATIONS)
    resto      = cantidad_necesaria  % len(AUGMENTATIONS)

    contador_global = len(os.listdir(clase_dir))  # continua numeracion tras las unicas

    for idx, (aug_fn, aug_name) in enumerate(zip(AUGMENTATIONS, AUG_NAMES)):
        n = por_metodo + (1 if idx < resto else 0)
        fuentes = random.choices(imagenes_src, k=n)
        for src in fuentes:
            img_pil = Image.open(src).convert("RGB")
            img_aug_pil = aug_fn(img_pil)

            # Segmentar la imagen augmentada antes de guardar
            img_cv2 = pil_to_cv2(img_aug_pil)
            segmentada, _ = segmentar_tomate_hsv(img_cv2)

            fname = f"unripe_{contador_global:04d}.jpg"
            cv2.imwrite(os.path.join(clase_dir, fname), segmentada)
            contador_global += 1

        print(f"    . {aug_name}: {n} imagenes generadas y segmentadas")

    total_aug = contador_global - (contador_global - cantidad_necesaria) - (contador_global - cantidad_necesaria - cantidad_necesaria + contador_global - cantidad_necesaria)
    print(f"    -> Total augmentadas para unripe: {cantidad_necesaria}")

# ─────────────────────────────────────────────
# 5. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────
def construir_dataset(dataset_path):
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    print("\nBuscando imagenes unicas por clase...")
    unicas = obtener_unicas_por_clase(dataset_path)

    for clase in CLASES:
        clase_dir = os.path.join(OUTPUT_DIR, clase)
        os.makedirs(clase_dir, exist_ok=True)
        imgs = unicas[clase]

        print(f"\nProcesando clase: {clase}")

        if clase == "unripe" and len(imgs) < TARGET_PER_CLASS:
            # Copiar y segmentar todas las unicas disponibles
            for src in imgs:
                fname = os.path.basename(src)
                dst = os.path.join(clase_dir, fname)
                copiar_y_segmentar(src, dst)

            faltantes = TARGET_PER_CLASS - len(imgs)
            print(f"  Solo hay {len(imgs)} unicas. Generando {faltantes} con augmentation...")
            aplicar_augmentation(imgs, clase_dir, faltantes)

        else:
            # Seleccionar 400 al azar, copiar y segmentar
            seleccionadas = random.sample(imgs, TARGET_PER_CLASS)
            for src in seleccionadas:
                fname = os.path.basename(src)
                dst = os.path.join(clase_dir, fname)
                copiar_y_segmentar(src, dst)

        total_en_carpeta = len(os.listdir(clase_dir))
        print(f"  {clase}: {total_en_carpeta} imagenes en carpeta")


if __name__ == "__main__":
  import kagglehub
  path = kagglehub.dataset_download("jasroop11/tomato-classification-unripe-ripe-and-damaged")
  print("Path to dataset files:", path)
  construir_dataset(path)