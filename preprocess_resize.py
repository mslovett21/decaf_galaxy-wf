import random
import glob
from PIL import Image



INPUT_DIR  = "galaxy_data/"
OUTPUT_DIR = "galaxy_data/"








def main():
    all_images = glob.glob(INPUT_DIR + "*.jpg")

    for img_path in all_images:
        img = Image.open(img_path)
        img = img.resize([224,224])
        img.save(img_path)






if __name__ == '__main__':
	main()


