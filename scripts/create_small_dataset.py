#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import pandas as pd
import os       
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob
from pathlib import Path
import sys
import argparse
from os import path
from IPython import embed
import time


DATA_DIR      = "galaxy_data/"








    
def main():

    f = open("galaxy_id_files.txt", "r")
    numbers = f.readlines()
    print(numbers.eval())
    embed()
    for num in numbers:
    	print(num)
    

    return 

if __name__ == '__main__':
    main()