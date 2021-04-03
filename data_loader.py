import glob, os
import random
import torch
from skimage import io, transform
from torch.utils.data import Dataset



class GalaxyDataset(Dataset):
    
    def __init__(self, filespath,prefix, transform = None):
        self.labels = {}
        self.filenames = []
        self.transform = transform

        
        class_0_files = glob.glob(filespath + prefix + "_class_0_*.jpg")
        class_1_files = glob.glob(filespath + prefix + "_class_1_*.jpg")
        class_2_files = glob.glob(filespath + prefix + "_class_2_*.jpg")
        class_3_files = glob.glob(filespath + prefix + "_class_3_*.jpg")
        class_4_files = glob.glob(filespath + prefix + "_class_4_*.jpg")

        self.filenames = class_0_files + class_1_files + class_2_files + class_3_files + class_4_files

        
        self.__datasetup__(class_0_files,0)
        self.__datasetup__(class_1_files,1)
        self.__datasetup__(class_2_files,2)
        self.__datasetup__(class_3_files,3)
        self.__datasetup__(class_4_files,4)
        
        random.shuffle(self.filenames)
    
    def __datasetup__(self,files, label):
        for filename in files:
            self.labels[filename] = label
            self.filenames.append(filename)


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        label = self.labels[filename]
        img = io.imread(filename)
        sample = {"image" : img,"label": label}
        
        if self.transform:
            sample = self.transform(sample)
        return sample


