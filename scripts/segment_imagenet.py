import os
import json
import argparse
import glob
import torch
import cv2
import numpy as np
import time
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm

def show_anns(anns, size=512):
    if len(anns) == 0:
        return
    img = np.ones((size, size, 3))
    for i, ann in enumerate(anns):
        if ann['area'] < 5000:
            continue
        m = ann['segmentation']
        m = mask_utils.decode(m)
        m = m.astype(bool)
        img[m] = mask_colormap[i % len(mask_colormap)]
    return img

def assign_images_to_gpu(gpu_id, image_paths, num_gpus):
    """
    Assigns images to a specific GPU based on its ID.
    
    :param gpu_id: Unique ID for the GPU (0 to total_gpus-1)
    :param total_gpus: Total number of GPUs available
    :param image_paths: List of paths to all images
    :return: List of paths assigned to the GPU
    """
    # Calculate the number of images per GPU
    images_per_gpu = len(image_paths) // num_gpus
    # Calculate start and end indices of images for this GPU
    start_idx = gpu_id * images_per_gpu
    end_idx = start_idx + images_per_gpu if gpu_id < num_gpus - 1 else len(image_paths)
    # Slice the list to get the paths assigned to this GPU
    return start_idx, end_idx


def calculate_centroid(mask):
    """Calculate the centroid of a binary mask."""
    y, x = np.where(mask == 1)
    centroid_x = np.mean(x)
    centroid_y = np.mean(y)
    return centroid_x, centroid_y

def sort_masks_by_segmentation_centerness(masks, image_width, image_height):
    """
    Sort masks based on the "centerness" of their segmentation.
    Closer to bottom-left is considered more central.

    :param masks: A list of masks, where each mask is a dict that includes a 'segmentation' key with a binary mask.
    :param image_width: Width of the original image.
    :param image_height: Height of the original image.
    :return: A list of masks sorted by their segmentation "centerness".
    """
    new_masks = []
    for mask in masks:
        centroid_x, centroid_y = calculate_centroid(mask['segmentation'])
        mask['centerness'] = np.sqrt((centroid_x - 0) ** 2 + (centroid_y - 0) ** 2)
        m = mask['segmentation']
        m = np.asfortranarray(m)
        m = mask_utils.encode(m)
        m['counts'] = str(m['counts'], encoding='utf-8')
        mask['segmentation'] = m
        new_masks.append(mask)
    sorted_masks = sorted(new_masks, key=lambda mask: mask['centerness'])
    return sorted_masks


def main(start, end, image_paths, sam_checkpoint):
    # Setup

    model_type = "vit_h"

    # Initialize SAM model
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.cuda()
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        # points_per_side=32,
        # pred_iou_thresh=0.86,
        # stability_score_thresh=0.92,
        # crop_n_layers=1,
        # crop_n_points_downscale_factor=2,
        # min_mask_region_area=5000,  # Requires open-cv to run post-processing
    )

    # start, end = assign_images_to_gpu(gpu_id, image_paths, num_gpus)
    with tqdm(total=end-start) as pbar:
        for idx in range(start, end):
            image = cv2.imread(image_paths[idx])  # Load image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (512, 512))
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                mask = mask_generator.generate(image)  # Generate mask
            # Store the processed mask in the dictionary
            mask_info = sort_masks_by_segmentation_centerness(mask, 256, 256)

            # Save to JSON
            with open(f"{image_paths[idx].replace('JPEG', 'json')}", 'w') as f:
                json.dump(mask_info, f)
            pbar.update(1)
            # with open(f"{image_paths[idx].replace('JPEG', 'json')}", 'r') as f:
            #     mask_info = json.load(f)
            # mask = show_anns(mask_info).astype(np.uint8)
            # mask = Image.fromarray(mask)
            # mask.save('mask.png')


def create_color_map(step=32):
    color_map = []
    for r in range(0, 256, step):
        for g in range(0, 256, step):
            for b in range(0, 256, step):
                color_map.append([r, g, b])
    return np.array(color_map)[1:]


mask_colormap = create_color_map()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate masks for images in a directory and store the results in a JSON file.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=1000)
    args = parser.parse_args()

    sam_checkpoint = "/home/mcg/VPA/scripts/sam_vit_h_4b8939.pth"  # wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    root = '/mnt/data/ImageNet2012'
    split = 'train'
    image_paths = sorted(glob.glob(os.path.join(root, split, "*", "*.JPEG")))[::-1]

    images_per_gpu = len(image_paths) // 1000
    start_idx = args.start * images_per_gpu
    end_idx = args.end * images_per_gpu if args.end != 1000 else len(image_paths)
    print(start_idx, end_idx)
    main(start_idx, end_idx, image_paths, sam_checkpoint)

    # with open(f"n01440764_97.json", 'r') as f:
    #     mask_info = json.load(f)
    # mask = show_anns(mask_info).astype(np.uint8)
    # mask = Image.fromarray(mask).convert('L')
    # print(mask)
    # mask.save('mask.png')
