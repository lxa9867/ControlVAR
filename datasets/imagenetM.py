import glob
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import json
from pycocotools import mask as mask_utils
import torch
from torch.nn import functional as F

def process_anns(anns, image_size, colormap):
    mask = np.zeros((image_size, image_size, 3))
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

class ImagenetMDataset(Dataset):
    def __init__(self, root: str, split: str = "train", transform=None, image_size=256,
                 v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16), separator=False, **kwargs):

        self.transforms = transform
        assert split == 'train'
        self.mask_paths = sorted(glob.glob(os.path.join(root, "train_mask/" "*", "*.json")))
        folder_paths = glob.glob(os.path.join(root, split, "*"))
        folders = [p.split('/')[-1] for p in folder_paths]
        self.cls = {k: v for v, k in enumerate(folders)}

        self.v_patch_nums = v_patch_nums
        self.image_size = image_size
        self.separator = separator
        self.colormap = create_color_map()
        print(f'ImagenetM dataset init: total images {len(self.mask_paths)}')


    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, index: int):
        mask_path = self.mask_paths[index]
        image_path = mask_path.replace('train_mask', 'train').replace('.json', '.JPEG')
        cls = self.cls[(image_path.split('/')[-2])]
        image = Image.open(image_path).convert('RGB')

        with open(mask_path, 'r') as f:
            mask_info = json.load(f)
        mask = process_anns(mask_info, 512, self.colormap).astype(np.uint8)  # 512 is fixed during the labelling
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

        sample = {'image': image, 'mask': mask, 'cls': cls, 'ignore_mask': ignore_masks, 'ignore_mask_': ignore_masks_, 'type': torch.tensor(0)}
        return sample


if __name__ == '__main__':
    image_root= '/voyager/ImageNet2012/train'
    mask_root = '/voyager/ImageNet2012/train_mask'

    mask_paths = sorted(glob.glob(os.path.join(mask_root, "*", "*.json")))[:10]
    print(len(mask_paths))
    fail_list = []
    from tqdm import tqdm
    colormap = create_color_map()
    with tqdm(total=len(mask_paths)) as pbar:
        for mask_path in mask_paths:
            size = os.stat(mask_path).st_size
            if size > 1000:
                try:
                    if not os.path.exists(mask_path.replace('train_mask', 'train').replace('.json', '.JPEG')):
                        print(mask_path)
                    with open(mask_path, 'r') as f:
                        mask_info = json.load(f)
                    mask = process_anns(mask_info, 512, colormap).astype(np.uint8)  # 512 is fixed during the labelling
                    mask = Image.fromarray(mask)
                    mask.save('mask.png')
                    print(mask.path)
                except:
                    fail_list.append(mask_path)
                    print(mask_path)

            pbar.update(1)
    # for path in fail_list:
    #     os.system(f'rm {path}')
