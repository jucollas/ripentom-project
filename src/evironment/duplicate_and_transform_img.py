from PIL import Image
from pathlib import Path 
from random import choice, randint

path = Path("./dataset_400/damaged")
dir_images = list(path.iterdir())
n_duplicates = 1

for i in range(n_duplicates):
  path_img =  choice(dir_images)
  img = Image.open(path_img)
  
  ran = randint(0, 3)
  new_img = img.rotate(90 * ran, expand=True)
  ran = randint(0, 1)
  if ran == 1:
    new_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)
  new_img.save(path_img.with_name(path_img.stem + '-duplex' + path_img.suffix))


