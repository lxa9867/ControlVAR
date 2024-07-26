import json
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from .color_map import mask_colormap
import glob
import os
from PIL import Image, ImageDraw
import numpy as np
from pycocotools import mask as mask_utils
from .mask_color import mask_colormap


def apply_color_map(id_map, color_list):
    # Convert the ID map to a NumPy array
    id_map_np = np.array(id_map)

    # Create an empty array to hold the colored image
    colored_image = np.zeros((id_map_np.shape[0], id_map_np.shape[1], 3), dtype=np.uint8)

    # Generate indices to directly map colors using advanced indexing
    color_indices = id_map_np % len(color_list)

    # Use advanced indexing to directly assign colors to the corresponding pixels
    colored_image[:, :, 0] = np.take(np.array(color_list)[:, 0], color_indices)
    colored_image[:, :, 1] = np.take(np.array(color_list)[:, 1], color_indices)
    colored_image[:, :, 2] = np.take(np.array(color_list)[:, 2], color_indices)

    return colored_image

class SA1BMaskDataset(Dataset):
    def __init__(self, root, transform):
        self.transforms = transform
        self.image_paths = sorted(glob.glob(os.path.join(root, "*", "*.jpg")))
        self.anno_paths = sorted(glob.glob(os.path.join(root, "*", "*.json")))
        self.colormap = mask_colormap

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        anno_path = self.anno_paths[idx]
        image = Image.open(image_path).convert('RGB')
        with open(anno_path) as f:
            annotations = json.load(f)['annotations']
        target = []

        for m in annotations:
            # decode masks from COCO RLE format
            m = mask_utils.decode(m['segmentation'])
            index = np.where(m == 1)
            r = np.sqrt(np.mean(index[0]) ** 2 + np.mean(index[1]) ** 2)
            target.append((r, m))

        target_ = [m for _, m in sorted(target)]
        if len(target_) != 0:
            target = np.argmax(np.stack(target_), axis=0)
        else:
            target = np.zeros((512, 512))
        mask = apply_color_map(target, self.colormap)
        mask = Image.fromarray(mask)

        if self.transforms:
            image, mask = self.transforms(image, mask)
        
        sample = {'image': image, 'mask': mask, 'cls': 0}  # cls will not be used

        return sample