import random, math

from typing import Tuple, Union

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode, ColorJitter

import numpy as np

from PIL import Image


class CustomTransform:
    def __call__(self, img: Image.Image, heatmap: np.ndarray, direction: int, **kargs):
        raise NotImplementedError(
            "CustomTransform is an abstract class. Please implement the __call__ method in a subclass."
        )


class CustomCompose(CustomTransform):
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(
        self, img: Image.Image, heatmap: np.ndarray, direction: int, **kargs
    ) -> Tuple[Image.Image, np.ndarray]:
        for transform in self.transforms:
            img, heatmap, direction = transform(img, heatmap, direction, **kargs)
        return img, heatmap, direction


class CustomRandomHorizontalFlip(CustomTransform):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self,
        img: Union[Image.Image, torch.Tensor],
        heatmap: np.ndarray,
        direction: int,
        **kargs
    ) -> Tuple[Image.Image, np.ndarray]:
        if random.random() < self.p:
            img = F.hflip(img)
            heatmap_torch = torch.from_numpy(heatmap.copy()).float()
            heatmap_torch = F.hflip(heatmap_torch)
            heatmap = heatmap_torch.numpy()

            # If left -> right, right -> left
            if direction == 0:
                direction = 1
            elif direction == 1:
                direction = 0

        return img, heatmap, direction


class CustomRandomVerticalFlip(CustomTransform):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self,
        img: Union[Image.Image, torch.Tensor],
        heatmap: np.ndarray,
        direction: int,
        **kargs
    ) -> Tuple[Image.Image, np.ndarray]:
        if random.random() < self.p:
            img = F.vflip(img)
            heatmap_torch = torch.from_numpy(heatmap.copy()).float()
            heatmap_torch = F.vflip(heatmap_torch)
            heatmap = heatmap_torch.numpy()

        return img, heatmap, direction


class CustomRandomRotation(CustomTransform):
    def __init__(self, degrees: int = 15):
        self.degrees = degrees

    def __call__(
        self, img: Union[Image.Image, torch.Tensor], heatmap: np.ndarray, direction
    ):
        angle = random.uniform(-self.degrees, self.degrees)
        img = F.rotate(img, angle, interpolation=Image.BILINEAR, fill=0)
        # Convert heatmap to a torch tensor for rotation
        heatmap_torch = torch.from_numpy(heatmap.copy()).float()
        heatmap_torch = F.rotate(
            heatmap_torch, angle, interpolation=InterpolationMode.NEAREST, fill=0
        )
        heatmap = heatmap_torch.numpy()

        return img, heatmap, direction


class CustomResize(CustomTransform):
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(
        self,
        img: Union[Image.Image, torch.Tensor],
        heatmap: np.ndarray,
        direction: int,
        **kargs
    ) -> Tuple[Image.Image, np.ndarray]:
        img = F.resize(img, self.size)
        return img, heatmap, direction


class CustomCrop(CustomTransform):
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(
        self,
        img: Union[Image.Image, torch.Tensor],
        heatmap: np.ndarray,
        direction: int,
        **kargs
    ) -> Tuple[Image.Image, np.ndarray]:
        img = F.crop(img, 0, 0, self.size[0], self.size[1])
        return img, heatmap, direction


class CustomRandomErase(CustomTransform):
    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: int = 0,
    ):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(
        self,
        img: Union[Image.Image, torch.Tensor],
        heatmap: np.ndarray,
        direction: int,
        **kargs
    ) -> Tuple[Image.Image, np.ndarray]:
        if random.random() < self.p:
            img_h, img_w = img.shape[-2:]
            area = img_h * img_w

            log_ratio = torch.log(torch.tensor(self.ratio))

            for _ in range(10):
                erase_area = (
                    area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
                )
                aspect_ratio = torch.exp(
                    torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
                ).item()

                h = int(round(math.sqrt(erase_area * aspect_ratio)))
                w = int(round(math.sqrt(erase_area / aspect_ratio)))
                if not (h < img_h and w < img_w):
                    continue

                i = random.randint(0, img_h - h)
                j = random.randint(0, img_w - w)

                img = F.erase(img, i, j, h, w, self.value)

                # Convert the dimensions to the heatmap dimensions and apply the erase
                heatmap_h, heatmap_w = heatmap.shape[1:]
                i = int(i * heatmap_h / img_h)
                j = int(j * heatmap_w / img_w)
                h = int(h * heatmap_h / img_h)
                w = int(w * heatmap_w / img_w)
                heatmap[:, i : i + h, j : j + w] = self.value

                return img, heatmap, direction

        return img, heatmap, direction


class CustomColorJitter(CustomTransform):
    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
    ):
        self.transform = ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def __call__(
        self,
        img: Union[Image.Image, torch.Tensor],
        heatmap: np.ndarray,
        direction: int,
        **kargs
    ) -> Tuple[Image.Image, np.ndarray]:
        img = self.transform(img)
        return img, heatmap, direction


class CustomNormalize(CustomTransform):
    def __init__(
        self, mean: Tuple[float, float, float], std: Tuple[float, float, float]
    ):
        self.mean = mean
        self.std = std

    def __call__(
        self,
        img: Union[Image.Image, torch.Tensor],
        heatmap: np.ndarray,
        direction: int,
        **kargs
    ) -> Tuple[Image.Image, np.ndarray]:
        img = F.normalize(img, self.mean, self.std)
        return img, heatmap, direction


class CustomToTensor(CustomTransform):
    def __call__(
        self,
        img: Union[Image.Image, torch.Tensor],
        heatmap: np.ndarray,
        direction: int,
        **kargs
    ) -> Tuple[torch.Tensor, np.ndarray]:
        img = F.to_tensor(img)
        return img, heatmap, direction
