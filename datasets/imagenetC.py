import glob
import random

import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import json
from pycocotools import mask as mask_utils
import torch
from torch.nn import functional as F
from tqdm import tqdm

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


def find_classes(directory):
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

class ImagenetCDataset(Dataset):
    def __init__(self, root: str, split: str = "train", transform=None, image_size=256,
                 v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16), separator=False, val_cond='depth', **kwargs):

        self.transforms = transform
        self.split = split
        self.load_dataset(root)
        classes, class_to_idx = find_classes(os.path.join(root, split))
        self.cond = {'mask': self.mask_paths, 'canny': self.canny_paths,
                     'depth': self.depth_paths, 'normal': self.normal_paths}
        self.cond_idx = {'mask': 0, 'canny': 1, 'depth': 2, 'normal': 3}
        self.class_to_idx = class_to_idx
        print('Use ImageFolder Class to IDX')
        self.v_patch_nums = v_patch_nums
        self.image_size = image_size
        self.separator = separator
        self.colormap = create_color_map()
        print(f'ImagenetC dataset init: total images '
              f'{max(len(self.mask_paths), len(self.canny_paths), len(self.depth_paths), len(self.normal_paths))}')
        if self.split == 'val':
            self.val_cond = val_cond
            print(f'Warning: Only use {self.val_cond} during the evaluation')

    def load_dataset(self, root):
        if 'ceph' in root:
            cond_info_path = os.path.join(root, f'{self.split}_cond_info_mpi.json')
        else:
            cond_info_path = os.path.join(root, f'{self.split}_cond_info.json')

        if os.path.exists(cond_info_path):
            print('load ImageNetC from json')
            with open(cond_info_path, 'r') as file:
                cond_info = json.load(file)
            self.mask_paths = cond_info['mask']
            self.canny_paths = cond_info['canny']
            self.depth_paths = cond_info['depth']
            self.normal_paths = cond_info['normal']
            print('mask, canny, depth, normal')
            print(len(self.mask_paths), len(self.canny_paths), len(self.depth_paths), len(self.normal_paths))
        else:
            print('load ImageNetC from glob')
            self.mask_paths = sorted(glob.glob(os.path.join(root, f"{self.split}_mask/" "*", "*.json")))
            self.canny_paths = sorted(glob.glob(os.path.join(root, f"{self.split}_canny/" "*", "*.jpeg")))
            self.depth_paths = sorted(glob.glob(os.path.join(root, f"{self.split}_depth/" "*", "*.jpeg")))
            self.normal_paths = sorted(glob.glob(os.path.join(root, f"{self.split}_normal/" "*", "*.jpeg")))
            colormap = create_color_map()
            for paths in [self.mask_paths, self.canny_paths, self.depth_paths, self.normal_paths]:
                with tqdm(total=len(paths)) as pbar:
                    for path in paths:
                        size = os.stat(path).st_size
                        pbar.update(1)
                        if size < 1000:
                            try:
                                if 'mask' in path:
                                    with open(path, 'r') as f:
                                        mask_info = json.load(f)
                                    mask = process_anns(mask_info, 512, colormap).astype(np.uint8)
                                    mask = Image.fromarray(mask)
                                else:
                                    mask = Image.open(path)
                            except:
                                print(path)
                                paths.remove(path)
            data = {
                'mask': self.mask_paths,
                'canny': self.canny_paths,
                'depth': self.depth_paths,
                'normal': self.normal_paths
            }
            with open(cond_info_path, 'w') as file:
                json.dump(data, file)


    def __len__(self):
        return max(len(self.mask_paths), len(self.canny_paths), len(self.depth_paths), len(self.normal_paths))

    def __getitem__(self, index: int):
        cond_type = random.choices(['mask', 'canny', 'normal', 'depth'], [0.25, 0.25, 0.25, 0.25], k=1)[0]
        # TODO: add mask val dataset
        if self.split == 'val':
            cond_type = self.val_cond
            # print(f'Warning: Only use {cond_type} during the evaluation')

        cond_path = self.cond[cond_type][index % len(self.cond[cond_type])]
        image_path = cond_path.replace(self.split+'_'+cond_type, self.split).replace('.json', '.JPEG').replace('.jpeg', '.JPEG')
        cls = self.class_to_idx[(image_path.split('/')[-2])]
        image = Image.open(image_path).convert('RGB')

        if cond_type == 'mask':
            with open(cond_path, 'r') as f:
                mask_info = json.load(f)
            mask = process_anns(mask_info, 512, self.colormap).astype(np.uint8)  # 512 is fixed during the labelling
            cond = Image.fromarray(mask)
        else:
            cond = Image.open(cond_path).convert('RGB')
        cond = cond.resize(image.size)

        if self.transforms:
            image, cond = self.transforms(image, cond)

        if cond_type == 'mask':
            ignore_mask = torch.ones_like(cond.sum(dim=0))

            ignore_mask[cond.sum(dim=0)==-3] = 0
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
        else:
            ignore_masks = torch.ones((1378,)) if self.separator else torch.ones((1360,))
            ignore_masks_ = torch.ones((1378,)) if self.separator else torch.ones((1360,))

        sample = {'image': image, 'mask': cond, 'cls': cls, 'ignore_mask': ignore_masks, 'ignore_mask_': ignore_masks_,
                  'type': torch.tensor(self.cond_idx[cond_type])}
        # print(cond_path)
        return sample


if __name__ == '__main__':
    root= '../ImageNet2012/'
    cond_info_path = os.path.join(root, 'cond_info.json')
    if os.path.exists(cond_info_path):
        print('load ImageNetC from json')
        with open(cond_info_path, 'r') as file:
            cond_info = json.load(file)
            mask_paths = cond_info['mask']
            canny_paths = cond_info['canny']
            depth_paths = cond_info['depth']
            normal_paths = cond_info['normal']

    from tqdm import tqdm
    with tqdm(total=len(normal_paths)) as pbar:
        for path in normal_paths:
            try:
                img = Image.open(path)#.convert('RGB')
            except:
                print(path)
            # img_path = path.replace('normal', 'train').replace('.json', '.JPEG').replace('.jpeg', '.JPEG')
            #     with open(mask_path, 'r') as f:
            #         mask_info = json.load(f)
            #     mask = process_anns(mask_info, 512, colormap).astype(np.uint8)  # 512 is fixed during the labelling
            #     mask = Image.fromarray(mask)
            #     mask.save('mask.png')
            #     print(mask.path)
            # except:
            #     fail_list.append(mask_path)
            #     print(mask_path)

            pbar.update(1)
    # for path in fail_list:
    #     os.system(f'rm {path}')
