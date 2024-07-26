import torch

def dice_coefficient(pred, target):
    smooth = 1e-5
    pred = pred.flatten(1)
    target = target.flatten(1)
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def dice_loss(pred, target):
    return 1 - dice_coefficient(pred, target)