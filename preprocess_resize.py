#!/usr/bin/env python3
import random
import glob
import sys
import argparse
from PIL import Image



def parse_args(args):
    parser = argparse.ArgumentParser(description="Enter description here")
    parser.add_argument(
        "-i","--input_dir",default="full_galaxy_dataset/",
        help="directory with data"
        )
    return parser.parse_args(args)



def main():
    args = parse_args(sys.argv[1:])
    input_dir  = args.input_dir

    all_images = glob.glob(input_dir + "*.jpg")

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


