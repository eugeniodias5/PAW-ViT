import os
import numpy as np
from PIL import Image

from data.dataset.AnimalDataset import AnimalDataset


class Yak(AnimalDataset):
    def __init__(self, root, mask_dir="", transform=None, mode="train"):
        super(Yak, self).__init__(root, mask_dir, transform, mode)
        self.imgs_path = os.path.join(root)

        if mode == "train":
            self.mask_path = os.path.join(root, "masks", mask_dir)
            txt_file = os.path.join(root, "train.txt")

            with open(txt_file, "r") as f:
                for line in f:
                    line = line.strip()
                    img_name, id, direction = line.split(" ")
                    self.imgs_name.append(img_name)
                    self.ids.append(int(id))
                    self.directions.append(int(direction))

        elif mode == "query":
            txt_file = os.path.join(root, "query.txt")

            with open(txt_file, "r") as f:
                for line in f:
                    line = line.strip()
                    img_name, id = line.split(" ")
                    self.imgs_name.append(img_name)
                    self.ids.append(int(id))

        elif mode == "gallery":
            txt_file = os.path.join(root, "gallery.txt")

            with open(txt_file, "r") as f:
                for line in f:
                    line = line.strip()
                    img_name, id = line.split(" ")
                    self.imgs_name.append(img_name)
                    self.ids.append(int(id))

        else:
            raise ValueError(
                "Invalid mode for Yak dataset. Choose between train, query, or gallery."
            )

        # Use dict to order ids from 0 to n-1
        self.ordered_id = {}
        for i, id in enumerate(set(self.ids)):
            self.ordered_id[id] = i

    def __getitem__(self, idx):
        img_path = os.path.join(self.imgs_path, self.imgs_name[idx])
        img = Image.open(img_path).convert("RGB")

        if self.mode == "train":
            img_path = os.path.join(self.imgs_path, self.imgs_name[idx])

            # Reading image
            mask_name = self.imgs_name[idx].replace(".jpg", ".npy")

            mask_path = os.path.join(self.mask_path, mask_name)
            mask = np.load(mask_path)

            id = int(self.ids[idx])
            direction = self.directions[idx]
            if self.transform:
                img, mask, direction = self.transform(img, mask, direction)

            return img, int(self.ordered_id[id]), mask, int(direction)

        else:
            if self.transform:
                img = self.transform(img)

            id = int(self.ids[idx])
            return img, self.ordered_id[id]
