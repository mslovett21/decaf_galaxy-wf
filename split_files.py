import os
import urllib.request
import shutil
import numpy as np
import zipfile
import random
import glob

INPUT_DIR  = "dataset/"
OUTPUT_DIR = "galaxy_data/"
#os.makedirs(OUTPUT_DIR)




def add_prefix(file_paths, prefix, output_dir):
    new_paths = []
    for fpath in file_paths:
        path, fname = fpath.split('/')
        fname = prefix + "_" + fname
        os.rename(fpath,output_dir+fname)
    return new_paths


def split_data_filenames(file_paths):
    random.shuffle(file_paths)
    train, val, test = np.split(file_paths, [int(len(file_paths)*0.7), int(len(file_paths)*0.8)])
    return train, val, test



def main():
    all_images = glob.glob(INPUT_DIR + "*.jpg")
    train, val, test = split_data_filenames(all_images)
    pf_train = add_prefix(train, "train",OUTPUT_DIR)
    pf_val = add_prefix(val, "val",OUTPUT_DIR)
    pf_test = add_prefix(test, "test",OUTPUT_DIR)





if __name__ == '__main__':
	main()


