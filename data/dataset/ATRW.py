import os, csv

from PIL import Image

import numpy as np
from torch.utils.data import Dataset


class ATRW(Dataset):
    def __init__(self, root, mask_dir="", transform=None, mode="train"):

        self.root = root
        self.transform = transform
        self.mode = mode

        self.imgs_name = []
        self.ids = []
        self.directions = []

        if mode == "train":
            self.imgs_path = os.path.join(root)
            self.mask_path = os.path.join(root, "masks", mask_dir)
            txt_file = os.path.join(root, "train.txt")

            with open(txt_file, "r") as f:
                for line in f:
                    line = line.strip()
                    img_name, id, direction = line.split(" ")
                    self.imgs_name.append(img_name)
                    self.ids.append(int(id))
                    self.directions.append(int(direction))

        elif mode == "test":
            self.imgs_path = os.path.join(root, "test")
            txt_file = os.path.join(root, "test.txt")

            with open(txt_file, "r") as f:
                for line in f:
                    line = line.strip()
                    img_name, _ = line.split(" ")
                    self.imgs_name.append(img_name)
                    img_name = int(img_name.split(".")[0])
                    self.ids.append(int(img_name))

        else:
            raise ValueError(
                "Invalid mode for ATRW dataset. Choose between train and test."
            )

    def __len__(self):
        return len(self.imgs_name)

    def __getitem__(self, idx):
        img_name = self.imgs_name[idx]
        img_path = os.path.join(self.imgs_path, img_name)
        img = Image.open(img_path).convert("RGB")
        id = int(self.ids[idx])

        if self.mode == "train":
            mask_path = os.path.join(
                self.mask_path, img_name.replace(".jpg", ".npy")
            )
            # Reading mask
            mask = np.load(mask_path)
            direction = int(self.directions[idx])

            if self.transform:
                img, mask, direction = self.transform(img, mask, direction)

            return img, id, mask, direction

        if self.transform:
            img = self.transform(img)

        return img, id

    def get_num_classes(self):
        return len(set(self.ids))

    def get_num_directions(self):
        return len(set(self.directions))
