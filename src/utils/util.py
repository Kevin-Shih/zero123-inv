import json
import numpy as np
import os
import torch
import random
from lovely_numpy import lo
from torchvision import transforms
from torch import Tensor
from PIL import Image
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import create_carvekit_interface, instantiate_from_config, load_and_preprocess
from transformers import AutoFeatureExtractor

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
    # print(mask.max(), mask.mean())
    return mask[0]

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_model_from_config(config, ckpt, device, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    # if 'global_step' in pl_sd:
    #     print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    if not isinstance(model, LatentDiffusion):
        raise TypeError("The instantiated model is not of type LatentDiffusion")
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    # model.train()
    model.eval()
    return model

def load_models(model_config_obj, path, device='cuda'):
    models = dict()
    print('Instantiating LatentDiffusion...', end='\r')
    models['turncam'] = load_model_from_config(model_config_obj, path, device=device, verbose=True)
    print('Instantiating Carvekit HiInterface...', end='\r')
    models['carvekit'] = create_carvekit_interface()

    print('Instantiating AutoFeatureExtractor...', end='\r')
    models['clip_fe'] = AutoFeatureExtractor.from_pretrained(
        'CompVis/stable-diffusion-safety-checker')
    return models

def load_image(models, input_im_path, preprocess=True, h=256, w=256, device='cuda'):
    '''
    :param input_im path.
    :return input_im (H, W, 3) array in [0, 1].
    '''
    input_im = Image.open(input_im_path)
    input_im.thumbnail([h, w], Image.Resampling.LANCZOS)

    old_size = input_im.size
    # start_time = time.time()

    if preprocess:
        input_im, forground_mask = load_and_preprocess(models['carvekit'], input_im)
        input_im = (input_im / 255.0).astype(np.float32)
        forground_mask[forground_mask >  0.5] = 1
        forground_mask[forground_mask <= 0.5] = 0
        # (H, W, 3) array in [0, 1].
    else:
        input_im = input_im.resize([h, w], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0
        # (H, W, 4) array in [0, 1].
        forground_mask = np.zeros([h, w], dtype=np.float32)
        forground_mask[input_im[:, :, -1] > 0.9] = [1.]
        alpha = input_im[:, :, 3:4]
        white_im = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white_im

        input_im = input_im[:, :, 0:3]
        # (H, W, 3) array in [0, 1].
    if old_size != input_im.shape[0:2]:
        print('old input_im:', lo(old_size))
        # print(f'Infer foreground mask (preprocess_image) took {time.time() - start_time:.3f}s.')
        print('new input_im:', lo(input_im))

    input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
    input_im = input_im * 2 - 1 # move to [-1, 1]
    input_im = transforms.Resize([h, w])(input_im)
    return input_im, Tensor(forground_mask).to(device)


def load_img_and_gt(
    models,
    image_dir: str = None,
    transform_fp: str = None,
    idx: int = 0,
    return_images=True,
    preprocess=True,
    device='cuda',
):
    """Load images from disk."""
    # if not transform_fp.startswith("/"):
    #     # allow relative path
    #     transform_fp = os.path.join(
    #         os.path.dirname(os.path.abspath(__file__)),
    #         "..",
    #         transform_fp,
    #     )

    with open(transform_fp, "r") as fp:
        meta = json.load(fp)

    frame = meta["frames"][idx]

    if return_images:
        fp = os.path.join(image_dir, frame["file_path"])
        img, mask = load_image(models, fp, preprocess=preprocess, device=device)

    c2w = torch.tensor(frame["transform_matrix"])
    pose = torch.tensor(frame["latlon"])

    return img, mask, c2w, pose

def split_list(lst, n):
    k, m = divmod(len(lst), n) 
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]