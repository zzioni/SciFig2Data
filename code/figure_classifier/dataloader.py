import pickle
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import io


class DocfigureDataset(Dataset):
    def __init__(self, root, labelnames, train = False, transforms = None):
        
        self.root = root
        
        self.labelnames = labelnames

        self.transforms = transforms
        
        self.train = train
        
        if self.train:
            with open(os.path.join(root, "annotation/train.txt"), "r") as f:
                self.data_files = f.readlines()
        else:
            with open(os.path.join(root, "annotation/test.txt"), "r") as f:
                self.data_files = f.readlines()

        self.images = []
        self.labels = []

        self.load_data()
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx]
        label = self.labels[idx]

        if self.transforms:
            image = self.transforms(image)

        return image, label


    def load_data(self):
        for file in self.data_files:
            img_fname, label = file.split(", ")
            img_file_path = os.path.join(self.root, "images/"+img_fname.strip())
            image = Image.open(img_file_path).convert('RGB')
            self.images.append(image)
            self.labels.append(self.labelnames.index(label.strip()))

        