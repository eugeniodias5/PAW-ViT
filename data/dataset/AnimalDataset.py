import os, csv

from PIL import Image

import numpy as np
from torch.utils.data import Dataset


# Create a Animal Dataset that will be parent class for all animal datasets
class AnimalDataset(Dataset):
    def __init__(self, root, mask_dir="", transform=None, mode="train"):
        self.root = root
        self.mask_dir = mask_dir
        self.transform = transform
        self.mode = mode

        self.imgs_name = []
        self.ids = []
        self.directions = []

    def __len__(self):
        return len(self.imgs_name)

    def get_num_classes(self):
        return len(set(self.ids))

    def get_num_directions(self):
        return len(set(self.directions))
