import os
import random

import numpy as np
import torch
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
from torchvision import datasets, transforms


from pycocotools.coco import COCO
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset
import os
from .color_map import mask_colormap


def calculate_centroid(polygons):
    """
    Calculate the centroid from multiple polygons.
    """
    total_x, total_y, total_points = 0, 0, 0
    for polygon in polygons:
        x_coordinates, y_coordinates = zip(*polygon)
        total_x += sum(x_coordinates)
        total_y += sum(y_coordinates)
        total_points += len(polygon)
    
    centroid_x = total_x / total_points
    centroid_y = total_y / total_points
    return centroid_x, centroid_y

def sort_annotations_by_centerness(annotations):
    """
    Sort annotations by the "centerness" of their masks considering all polygons.
    """
    centroids = []
    for ann in annotations:
        if 'segmentation' in ann:
            all_polygons = []
            for segment in ann['segmentation']:
                # Convert segmentation format to list of (x, y) tuples for each polygon
                polygon = [(segment[i], segment[i+1]) for i in range(0, len(segment), 2)]
                all_polygons.append(polygon)
            if all_polygons:
                centroid = calculate_centroid(all_polygons)
                centroids.append((centroid, ann))
    
    # Sort by y (descending) and then x (ascending)
    sorted_annotations = [ann for _, ann in sorted(centroids, key=lambda x: (-x[0][1], x[0][0]))]
    return sorted_annotations



def apply_mask_and_propagate_colors(image, mask, iterations=64, kernel_size=(13, 13)):
    """
    Apply a mask as the alpha channel to an image and propagate edge colors into transparent areas.
    
    Args:
        image (PIL.Image): The original image, assumed to be in RGB.
        mask (PIL.Image): The mask to be used as the alpha channel.
        iterations (int): Number of iterations for color propagation.
        kernel_size (tuple): The kernel size for the Gaussian filter.
        
    Returns:
        PIL.Image: The modified image with propagated colors.
    """
    # Convert PIL images to numpy arrays
    image_array = np.array(image.convert('RGB'))
    mask_array = np.array(mask)

    # Ensure mask is in the correct format (single channel, same size as image)
    if mask_array.ndim == 3:
        mask_array = cv2.cvtColor(mask_array, cv2.COLOR_BGR2GRAY)
    elif mask_array.shape != image_array.shape[:2]:
        raise ValueError("Mask must be the same size as the image.")

    # Combine RGB image and mask to form an RGBA image
    image_rgba = np.dstack((image_array, mask_array))

    # Apply Gaussian filter to propagate colors
    for _ in range(iterations):
        for c in range(3):  # Apply the filter to each RGB channel
            channel = image_rgba[:, :, c]
            channel_blurred = cv2.GaussianBlur(channel, kernel_size, 0)
            # Only update pixels where alpha is 0 (transparent areas)
            channel_updated = np.where(mask_array != 0, channel_blurred, channel)
            image_rgba[:, :, c] = channel_updated
    
    # Convert the modified RGBA numpy array back to a PIL Image
    modified_image = Image.fromarray(image_rgba, 'RGBA').convert("RGB")

    return modified_image


class MSCOCOMaskDataset(Dataset):
    def __init__(self, annotation_path, img_dir, image_size, transform=None, min_area=3000):
        """
        Args:
            annotation_path (string): Path to the MSCOCO annotation file.
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            min_area (int): Minimum area to consider an object as large.
        """
        self.coco = COCO(annotation_path)
        self.img_dir = img_dir
        if transform is None:
            self.transforms = transforms.Compose([
                transforms.Resize((image_size, image_size)),  
                transforms.ToTensor()
                ])
        else:
            self.transforms = transform
        self.ids = sorted(self.coco.imgs.keys())
        self.min_area = min_area
        # for img_id in sorted(self.coco.imgs.keys()):
        #     ann_ids = self.coco.getAnnIds(imgIds=img_id, areaRng=[min_area, float('inf')])
        #     for ann_id in ann_ids:
        #         ann = self.coco.loadAnns(ann_id)[0]
        #         if ann['area'] >= min_area:  # Ensure the annotation meets the area requirement
        #             self.ids.append((img_id, ann_id))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]  # Unpack the tuple

        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.img_dir, path)).convert('RGB')

        ann_ids = self.coco.getAnnIds(imgIds=img_id, areaRng=[self.min_area, float('inf')], iscrowd=0)

        annotations = self.coco.loadAnns(ann_ids)

        annotations = sort_annotations_by_centerness(annotations)

        mask = Image.new('RGB', image.size, (0, 0, 0))

        for index, ann in enumerate(annotations):
            # if ann['iscrowd'] == 1:
            #     binary_mask = self.coco.annToMask(ann)
            #     mask = Image.fromarray(binary_mask * 255).convert('RGB')
            #     break
            # For polygonal annotations (COCO's 'segmentation' format)
            for seg in ann['segmentation']:
                draw = ImageDraw.Draw(mask)
                # Assuming 'seg' is a flat list of [x1,y1,x2,y2,...], convert to [(x1,y1),(x2,y2),...]
                polygon = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
                draw.polygon(polygon, fill=tuple(mask_colormap[index+1]))


        if self.transforms:
            image = self.transforms(image)
            mask = self.transforms(mask)
        
        sample = {'image': image, 'mask': mask}

        return sample
