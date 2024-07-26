import os
from PIL import Image
import glob

root = ['cfg_1.5_1.5_1.5_canny', 'cfg_2.0_2.0_2.0_canny',
        'cfg_2.5_2.5_2.5_canny', 'cfg_3.0_3.0_3.0_canny',
        'cfg_3.5_3.5_3.5_canny', 'cfg_4.0_4.0_4.0_canny',
        'canny', 'depth', 'normal', 'mask']
paths = glob.glob('/home/cmu/xiangli/experiments/d30/cfg_2.0_2.0_2.0_canny/*/*.png')
for path in paths:
    combined_img = Image.new("RGB", (256 * 10, 256))
    img_paths = [path.replace('cfg_2.0_2.0_2.0_canny', p) for p in root]
    print(img_paths)
    for i, img_path in enumerate(img_paths):
        img = Image.open(img_path)
        combined_img.paste(img, (256 * i, 0))
    save_path = path.replace('cfg_2.0_2.0_2.0_canny', 'combine_canny')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    combined_img.save(save_path)