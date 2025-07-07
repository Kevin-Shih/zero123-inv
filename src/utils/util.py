import numpy as np
from torchvision import transforms

def mask_resize(mask, size:int):
    """
    Resize a mask to the specified size.
    Args:
        mask (torch.Tensor): The input mask tensor.
        size (tuple): The target size (height, width).
    Returns:
        torch.Tensor: The resized mask tensor.
    """
   
    resize_transform = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
    mask = resize_transform(mask.unsqueeze(0))
    mask = transforms.CenterCrop((256, 256))(mask)
    return mask[0]