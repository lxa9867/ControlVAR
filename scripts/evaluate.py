import os 
import json
import argparse
import json
import yaml
from glob import glob
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm 
from glob import glob
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision import transforms
# from utils import str2bool



class ImageListDataset(torch.utils.data.Dataset):
    def __init__(self, image_list, transform=None):
        
        self.transform = transform
        
        filtered_image_list = []
        # go through
        for image_path in image_list:
            try:
                img = Image.open(image_path).convert("RGB")
            except:
                print("Error loading image", image_path)
                continue
            filtered_image_list.append(image_path)
        self.image_list = filtered_image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img = Image.open(self.image_list[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='/media/Bootes/datasets/imagenet/val', help='generated image folder')
    parser.add_argument('--gt_dir', type=str, default='/media/Bootes/datasets/imagenet/val', help='grond truth imagenet folder')
    parser.add_argument('--save_dir', type=str, default='./tmp', help='grond truth imagenet folder')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--image_size', type=int, default=256, help='image size')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    parser.add_argument('--inception',  action='store_true', default=False)
    parser.add_argument('--psnr', action='store_true', default=False)
    parser.add_argument('--ssim', action='store_true', default=False)
    parser.add_argument('--lpips',  action='store_true', default=False)
    
    opt = parser.parse_args()

    # compute_clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").cuda()
    # compute_fid_score = FrechetInceptionDistance(normalize=True).cuda()
    if opt.inception:
        compute_is_score = InceptionScore()#.cuda()
    if opt.psnr:
        compute_psnr_score = PeakSignalNoiseRatio().cuda()
    if opt.ssim:
        compute_ssim_score = StructuralSimilarityIndexMeasure().cuda()
    if opt.lpips:
        compute_lpips_score = LearnedPerceptualImagePatchSimilarity('vgg').cuda()
        
    
    img_transform = transforms.Compose([
        # transforms.Resize((opt.image_size, opt.image_size)),
        transforms.Resize(opt.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(opt.image_size),
        transforms.ToTensor(),
    ])
    
    gt_files = sorted(glob(os.path.join(opt.gt_dir, '*', '*.JPEG')))
    gen_files = sorted(glob(os.path.join(opt.img_dir, '*', '*.png')))
    assert len(gt_files) == len(gen_files)
    gt_dataset = ImageListDataset(gt_files, transform=img_transform)
    gen_dataset = ImageListDataset(gen_files, transform=img_transform)
    gt_dataloader = DataLoader(gt_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)
    gen_dataloader = DataLoader(gen_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)
    
    for gt_imgs, gen_imgs in tqdm(zip(gt_dataloader, gen_dataloader)):
        gt_imgs = gt_imgs#.cuda()
        gen_imgs = gen_imgs#.cuda()
        
        if opt.lpips:
            # lpips need images in [-1 ,1]
            compute_lpips_score.update(gt_imgs, gen_imgs)
        
        gt_imgs *= 255.0
        gen_imgs *= 255.0


        if opt.psnr:
            compute_psnr_score.update(gen_imgs, gt_imgs)
        
        if opt.ssim:
            compute_ssim_score.update(gen_imgs, gt_imgs)
        

        gt_imgs = gt_imgs.to(torch.uint8)
        # compute_fid_score.update(gt_imgs, real=True)
        gen_imgs = gen_imgs.to(torch.uint8)
        # compute_fid_score.update(gen_imgs, real=False)
        if opt.inception:
            compute_is_score.update(gen_imgs)


    results = {}
    # fid_score = compute_fid_score.compute().detach()
    # fid_score = round(float(fid_score), 8)
    # results['fid'] = fid_score
    # print(f"img_dir {opt.img_dir}, fid {fid_score}")
    
    if opt.lpips:
        lpips_score = compute_lpips_score.compute().detach()
        lpips_score = round(float(lpips_score), 8)
        results['lpips'] = lpips_score
        print(f"img_dir {opt.img_dir}, lpips {lpips_score}")    
    if opt.psnr:
        psnr_score = compute_psnr_score.compute().detach()
        psnr_score = round(float(psnr_score), 8)
        results['psnr'] = psnr_score
        print(f"img_dir {opt.img_dir}, psnr {psnr_score}")
    if opt.ssim:
        ssim_score = compute_ssim_score.compute().detach()
        ssim_score = round(float(ssim_score), 8)
        results['ssim'] = ssim_score
        print(f"img_dir {opt.img_dir}, ssim {ssim_score}")
    if opt.inception:
        is_score = compute_is_score.compute()[0].detach()
        is_score = round(float(is_score), 8)
        results['is'] = is_score
        print(f"img_dir {opt.img_dir}, is {is_score}")
    
    os.makedirs(opt.save_dir, exist_ok=True)
    with open(os.path.join(opt.save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    
    # # save numpy array for evaluation
    # all_gen_imgs_array = np.concatenate(all_gen_imgs_array, axis=0)
    # all_gt_imgs_array = np.concatenate(all_gt_imgs_array, axis=0)
    # np.savez('gen_imgs', all_gen_imgs_array)
    # np.savez('gt_imgs', all_gt_imgs_array)
    

if __name__ == '__main__':
    main()