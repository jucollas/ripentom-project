import os, shutil, random
from pathlib import Path



def split_database(path_base, path_dest):
    CLASES = ["damaged", "ripe", "unripe",]
    SPLIT  = 0.70
    random.seed(42)
    print("HAAAA")

    for nombre_clase in CLASES:
        imgs = list((path_base / nombre_clase).glob("*.*"))
        print(imgs)
        random.shuffle(imgs)
        n_train = int(len(imgs) * SPLIT)

        for split, subset in [("train", imgs[:n_train]), ("val", imgs[n_train:])]:
            dest_dir = path_dest / split / nombre_clase
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img in subset:
                shutil.copy(img, dest_dir / img.name)

        print(f"{nombre_clase}: {n_train} train | {len(imgs)-n_train} val")

    print(f"\n✓ Dataset dividido en {path_dest}")
    
if __name__ == "__main__":
    path_base = Path("data/raw/")
    path_dest = Path("data/processed/")
    split_database(path_base, path_dest)
    
    
