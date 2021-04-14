import random
import glob
from PIL import Image


INPUT_DIR  = "galaxy_data/"



#INPUT_DIR  = "final_galaxy_dataset/"
OUTPUT_DIR = INPUT_DIR








def main():
    all_images = glob.glob(INPUT_DIR + "*.jpg")

    for img_path in all_images:
        img = Image.open(img_path)
        width, height = img.size
        new_width = 256
        new_height = 256

        left   = (width - new_width)/2
        top    = (height - new_height)/2
        right  = (width + new_width)/2
        bottom = (height + new_height)/2

        img = img.crop((left, top, right, bottom))
        img.save(img_path)






if __name__ == '__main__':
	main()


