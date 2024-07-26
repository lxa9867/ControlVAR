import cv2
import numpy as np
from PIL import Image
import torch, io
import torchdata.datapipes as dps
from braceexpand import braceexpand
from .mask_color import mask_colormap
from pycocotools import mask as mask_utils

def calculate_centroid_poly(polygons):
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
                polygon = [(segment[i], segment[i + 1]) for i in range(0, len(segment), 2)]
                all_polygons.append(polygon)
            if all_polygons:
                centroid = calculate_centroid_poly(all_polygons)
                centroids.append((centroid, ann))

    # Sort by y (descending) and then x (ascending)
    sorted_annotations = [ann for _, ann in sorted(centroids, key=lambda x: (-x[0][1], x[0][0]))]
    return sorted_annotations

def tensor_encoder(obj):
    """Custom encoder for PyTorch tensors."""
    if isinstance(obj, torch.Tensor):
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        return buffer.getvalue()  # Return tensor serialized as bytes
    return obj  # Fallback for types this encoder doesn't handle


def decode_pkl(item):
    key, value = item
    value_file_obj = value.file_obj
    if key == '__key__':
        return key, value
    elif key.endswith('.code'):
        return key, torch.load(value_file_obj, map_location='cpu')
    elif key.endswith('.txt'):
        return key, {'text': value_file_obj.read().decode('utf-8')}
    else:
        return key, value_file_obj.read().decode('utf-8')


def unwarp_data(item):
    unwarpped = {}
    for key, value in item.items():
        if isinstance(value, dict):
            unwarpped.update(value)
        elif value is not None:
            unwarpped[key] = value
    if '__key__' in unwarpped and '/' in unwarpped['__key__']:
        unwarpped['__key__'] = unwarpped['__key__'].split('/')[-1]
    return unwarpped


def build_datapipe(data_dir,
                   masks='*.tar',
                   decode_fn=None,
                   max_length=1024,
                   reverse_ratio=0.5,
                   cycle_count=None,
                   batch_size=None,
                   shuffle=True,
                   recursive=True,
                   non_deterministic=False):
    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))

    datapipe = dps.iter.FileLister(data_dir, masks=masks, recursive=recursive, non_deterministic=non_deterministic)
    # cycle time of the dataset
    datapipe = datapipe.cycle(count=cycle_count)
    # shuffle
    if shuffle:
        datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='b')
    # TODO: use this according to decode_fn
    datapipe = datapipe.load_from_tar()

    # data processing and decoding map
    if decode_fn is not None:
        datapipe = datapipe.map(decode_fn)
    # streaming as dict
    datapipe = datapipe.webdataset()
    # unwrap the data
    datapipe = datapipe.map(unwarp_data)
    # filter data if necessary
    # datapipe = datapipe.filter(filter_data_for_llm)

    # shuffle with buffer size
    if shuffle:
        datapipe = datapipe.shuffle(buffer_size=4096)

    # batch data
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        datapipe = datapipe.collate()
    return datapipe


def calculate_centroid(label_image, label):
    """Calculate the centroid of a single labeled component."""
    rows, cols = np.where(label_image == label)
    if len(rows) == 0:
        return None
    centroid_x = np.mean(cols)
    centroid_y = np.mean(rows)
    return centroid_x, centroid_y


def semantic_to_instance_map(semantic_map_path):
    # Load the semantic map image
    semantic_map = np.array(Image.open(semantic_map_path))

    # Create the category mask: every non-black pixel is part of the category
    category_mask = np.any(semantic_map != [0, 0, 0], axis=-1).astype(np.uint8) * 255

    # Find connected components (individual instances) within the category mask
    num_labels, labels_im = cv2.connectedComponents(category_mask)

    # Collect centroids for sorting
    centroids = []
    for label in range(1, num_labels):  # Skip the background
        centroid = calculate_centroid(labels_im, label)
        if centroid:
            centroids.append((label, centroid))

    # Sort centroids based on their distance from the bottom-right corner
    # The image's bottom-right corner is at (max_x, max_y)
    max_x, max_y = labels_im.shape[1], labels_im.shape[0]
    centroids.sort(key=lambda x: -(x[1][0] + x[1][1]))  # Sort by the sum of x and y coordinates

    # Create a visual representation (optional)
    instance_map_visual = np.zeros(semantic_map.shape, dtype=np.uint8)
    for idx, (label, _) in enumerate(centroids, start=1):
        color = mask_colormap[idx]
        instance_map_visual[labels_im == label] = color

    # Convert the visual instance map to a PIL image and save

    # print(f"Found {num_labels - 1} instances, sorted")
    return Image.fromarray(instance_map_visual)