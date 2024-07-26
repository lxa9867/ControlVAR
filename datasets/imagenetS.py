import glob
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageNet
import os
from PIL import Image
from .utils import semantic_to_instance_map
import torch

class ImagenetSDataset(Dataset):
    def __init__(self, root: str, split: str = "train-semi", transform=None, image_size=256, **kwargs):
        if transform is None:
            self.transforms = transforms.Compose([
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transforms = transform
        self.image_paths = sorted(glob.glob(os.path.join(root, split, "*", "*.JPEG")))
        self.mask_paths = sorted(glob.glob(os.path.join(root, f"{split}-segmentation", "*", "*.png")))
        folder_paths = glob.glob(os.path.join(root, split, "*"))
        self.cls = [p.split('/')[-1] for p in folder_paths]
        print(f'Imagenet dataset init: total images {len(self.image_paths)}')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        # index = 1
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        cls = self.cls.index(image_path.split('/')[-2])
        image = Image.open(image_path).convert('RGB')
        mask = semantic_to_instance_map(mask_path)

        if self.transforms:
            image, mask = self.transforms(image, mask)
        sample = {'image': image, 'mask': mask, 'cls': cls, 'ignore_mask': torch.ones_like(mask)}
        return sample