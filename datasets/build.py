import torch 
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from .coco import MSCOCOMaskDataset
from .sa1b import SA1BMaskDataset
from .imagenetS import ImagenetSDataset
from .imagenetC import ImagenetCDataset
from .imagenetM import ImagenetMDataset
from .entityS import EntitySegDataset
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset
from .transforms_image import create_image_mask_transforms


def create_transforms(image_size):
    
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])



def create_dataset(dataset_name, args, split='train'):
    
    if dataset_name == "imagenet":

        dataset = ImageFolder(args.data_dir, transform=create_transforms(args.image_size))
    
    elif dataset_name == "coco":

        dataset = MSCOCOMaskDataset(args)

    elif dataset_name == "SA1B":
        assert args.uncond, 'must be uncond generation'
        dataset = SA1BMaskDataset(args.data_dir, create_image_mask_transforms(args.image_size))

    elif dataset_name == "imagenetS":
        dataset_train = ImagenetSDataset(args.data_dir, split='train-semi', image_size=(args.image_size, args.image_size),
                                         transform=create_image_mask_transforms(args.image_size))
        dataset_val = ImagenetSDataset(args.data_dir, split='validation', image_size=args.image_size,
                                         transform=create_image_mask_transforms(args.image_size))
        dataset = ConcatDataset([dataset_train, dataset_val])

    elif dataset_name == "imagenetM":
        dataset = ImagenetMDataset(args.data_dir, split='train', image_size=args.image_size,
                                         transform=create_image_mask_transforms(args.image_size, True),
                                         v_patch_nums=args.v_patch_nums, separator=args.separator,)
    elif dataset_name == "imagenetC":
        dataset = ImagenetCDataset(args.data_dir, split=split, image_size=args.image_size,
                                         transform=create_image_mask_transforms(args.image_size, split=='train'),
                                         v_patch_nums=args.v_patch_nums, separator=args.separator, val_cond=args.val_cond)

    elif dataset_name == "entityS":
        dataset = EntitySegDataset(args.data_dir, split='train', image_size=args.image_size,
                                   transform=create_image_mask_transforms(args.image_size, True),
                                   v_patch_nums=args.v_patch_nums, separator=args.separator, )

    else:
        raise NotImplementedError

    return dataset
        
        