import glob
import numpy as np
import cv2
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import torchvision.transforms as transforms
import os
from PIL import Image
import json
from pycocotools import mask as mask_utils
import torch
from torch.nn import functional as F


def process_anns(anns, image, colormap):
    mask = np.zeros_like(image)
    for i, ann in enumerate(anns):
        if ann['area'] < 5000:
            continue
        m = ann['segmentation']
        m = mask_utils.decode(m)
        X, Y = m.shape[1], m.shape[0]
        index = np.where(m == 1)
        x = int(np.mean(index[1]) // (X / 11))
        y = int(np.mean(index[0]) // (Y / 11))
        m = m.astype(bool)
        assert x * y < 124
        mask[m] = colormap[(x * y) % len(colormap)]
    return mask

def create_color_map():
    color_map = []
    for r in [0, 64, 128, 192, 255]:
        for g in [0, 64, 128, 192, 255]:
            for b in [0, 64, 128, 192, 255]:
                color_map.append([r, g, b])
    return np.array(color_map)[1:]

class EntitySegDataset(Dataset):
    def __init__(self, root: str, split: str = "train", transform=None, image_size=256,
                 v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16), separator=False, **kwargs):
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
        assert split == 'train'
        self.coco = COCO(os.path.join(root, f"entityseg_{split}.json"))
        self.img_dir = os.path.join(root, 'images')

        self.v_patch_nums = v_patch_nums
        self.image_size = image_size
        self.separator = separator
        self.colormap = create_color_map()
        self.ids = sorted(self.coco.imgs.keys())
        print(f'EntitySegmentation dataset init: total images {len(self.ids)}')


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]  # Unpack the tuple

        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = cv2.imread(os.path.join(self.img_dir, path), cv2.COLOR_BGR2RGB).astype(np.uint8)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        mask = process_anns(annotations, image, self.colormap).astype(np.uint8)
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        
        if self.transforms:
            image, mask = self.transforms(image, mask)

        ignore_mask = torch.ones_like(mask.sum(dim=0))

        ignore_mask[mask.sum(dim=0)==-3] = 0
        ignore_masks = []
        ignore_masks_ = []
        for si, pm in enumerate(self.v_patch_nums):
            num_sp_tokens = 1 if (si != 0 and self.separator) else 0
            if si < 5:  # [1, 2, 3, 4, 5, 6,]
                ignore_masks.append(torch.ones((pm ** 2 + num_sp_tokens,)))  # mask ignore
                ignore_masks.append(torch.ones((pm ** 2 + num_sp_tokens,)))  # image ignore

                ignore_masks_.append(torch.ones((pm ** 2 + num_sp_tokens,)))  # image ignore
                ignore_masks_.append(torch.ones((pm ** 2 + num_sp_tokens,)))  # mask ignore

            else:
                ignore_mask_ = F.interpolate(ignore_mask[None, None, :, :], (pm, pm), mode='nearest')\
                    .permute((0, 2, 3, 1)).reshape((-1,))
                if self.separator:
                    ignore_mask_ = torch.concat((torch.ones(1,), ignore_mask_), dim=0)

                ignore_masks.append(ignore_mask_)  # mask ignore
                ignore_masks.append(torch.ones((pm ** 2 + num_sp_tokens,)))  # image ignore

                ignore_masks_.append(torch.ones((pm ** 2 + num_sp_tokens,)))  # image ignore
                ignore_masks_.append(ignore_mask_)  # mask ignore

        ignore_masks = torch.concat(ignore_masks, dim=0)
        ignore_masks_ = torch.concat(ignore_masks_, dim=0)

        sample = {'image': image, 'mask': mask, 'cls': 1000, 'ignore_mask': ignore_masks, 'ignore_mask_': ignore_masks_}
        return sample


if __name__ == '__main__':
    root= '/voyager/entityseg/'
    split = 'train'
    dataset = EntitySegDataset(root, split, transform=create_image_mask_transforms(256))
    dataset[0]
    
    # for path in fail_list:
    #     os.system(f'rm {path}')