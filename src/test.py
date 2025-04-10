from math import e
import os
# import math
# from cv2 import mean
import fire
import argparse
import numpy as np
import time
import re

import torch
from contextlib import nullcontext
# from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
# from einops import rearrange
# from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import create_carvekit_interface, load_and_preprocess, instantiate_from_config
from lovely_numpy import lo
from omegaconf import OmegaConf
from PIL import Image
from rich import print
from torch import Tensor, optim, nn
from torch.nn.parameter import Parameter
from torch.amp.autocast_mode import autocast
from torch.utils.tensorboard import writer
from torchvision import transforms
from transformers import AutoFeatureExtractor

def load_model_from_config(config, ckpt, device, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    if not isinstance(model, LatentDiffusion):
        raise TypeError("The instantiated model is not of type LatentDiffusion")
    m, u = model.load_state_dict(sd, strict=False)
    # print('model type:', type(model))
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.train()
    # print(type(model))
    # # for p in model.parameters():
    #     # p.requires_grad = False
    return model

def preprocess_image(models, input_im, preprocess, h=256, w=256, device='cuda'):
    '''
    :param input_im (PIL Image).
    :return input_im (H, W, 3) array in [0, 1].
    '''
    old_size = input_im.size
    start_time = time.time()

    if preprocess:
        input_im = load_and_preprocess(models['carvekit'], input_im)
        input_im = (input_im / 255.0).astype(np.float32)
        # (H, W, 3) array in [0, 1].
    else:
        input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0
        # (H, W, 4) array in [0, 1].

        # old method: thresholding background, very important
        # input_im[input_im[:, :, -1] <= 0.9] = [1., 1., 1., 1.]

        # new method: apply correct method of compositing to avoid sudden transitions / thresholding
        # (smoothly transition foreground to white background based on alpha values)
        alpha = input_im[:, :, 3:4]
        white_im = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white_im

        input_im = input_im[:, :, 0:3]
        # (H, W, 3) array in [0, 1].
    if old_size != input_im.shape[0:2]:
        print('old input_im:', lo(old_size))
        print(f'Infer foreground mask (preprocess_image) took {time.time() - start_time:.3f}s.')
        print('new input_im:', lo(input_im))
    else:
        print(f'Infer foreground mask (preprocess_image) took {time.time() - start_time:.3f}s.')
        # print('input_im:', lo(input_im))
    input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
    input_im = input_im * 2 - 1 # ??
    input_im = transforms.Resize([h, w])(input_im)
    return input_im

def calculate_param(LDModel, step, n_samples, device):
    alphas = LDModel.alphas_cumprod
    alphas_prev = LDModel.alphas_cumprod_prev
    sqrt_one_minus_alphas = LDModel.sqrt_one_minus_alphas_cumprod
    sigmas = LDModel.ddim_sigmas_for_original_steps

    # select parameters corresponding to the currently considered timestep
    a_t = torch.full((n_samples, 1, 1, 1), alphas[step], device=device)
    a_prev = torch.full((n_samples, 1, 1, 1), alphas_prev[step], device=device)
    sigma_t = torch.full((n_samples, 1, 1, 1), sigmas[step], device=device)
    sqrt_one_minus_at = torch.full((n_samples, 1, 1, 1), sqrt_one_minus_alphas[step],device=device)    
    return a_t, a_prev, sigma_t, sqrt_one_minus_at

def sample_model(input_im, target_im, LDModel, precision, h, w,
                 n_samples, scale, ddim_eta,
                 elevation, azimuth, radius, step=651):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with LDModel.ema_scope():
            # region
            #noisy_latent = LDModel.prepare_latent(target_im, t)
            img_cond = LDModel.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            target_encode = LDModel.encode_first_stage(target_im)
            target_im_z = LDModel.get_first_stage_encoding(target_encode)
            T = torch.cat([elevation, torch.sin(azimuth), torch.cos(azimuth), radius])
            T_batch = T[None, None, :].repeat(n_samples, 1, 1)
            c = torch.cat([img_cond, T_batch], dim=-1)
            c_proj = LDModel.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c_proj]
            cond['c_concat'] = [LDModel.encode_first_stage((input_im)).mode().repeat(n_samples, 1, 1, 1)]
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8, device=img_cond.device)]
                uc['c_crossattn'] = [torch.zeros_like(c_proj, device=img_cond.device)]
            else:
                uc = None

            t = torch.full((n_samples,), step, device=img_cond.device, dtype=torch.long)
            size = (n_samples, 4, h // 8, w // 8)
            img = torch.randn(size, device=img_cond.device)
            # endregion

            # print(f'conditioning_key= {LDModel.model.conditioning_key}')
            # print(f"target_im_z={target_im_z.shape}, target_latent={target_latent.shape}")
            if uc is None or scale == 1.:
                e_t = LDModel.apply_model(img, t, cond)
            else:
                x_in = torch.cat([img] * 2)
                t_in = torch.cat([t] * 2)
                if isinstance(cond, dict):
                    assert isinstance(uc, dict)
                    c_in = dict()
                    for k in cond:
                        if isinstance(cond[k], list):
                            c_in[k] = [torch.cat([
                                uc[k][i],
                                cond[k][i]]) for i in range(len(cond[k]))]
                        else:
                            c_in[k] = torch.cat([
                                    uc[k],
                                    cond[k]])
                else:
                    assert not isinstance(uc, dict)
                    c_in = torch.cat([uc, cond])
                e_t_uncond, e_t = LDModel.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + scale * (e_t - e_t_uncond)


            a_t, a_prev, sigma_t, sqrt_one_minus_at = calculate_param(LDModel, step, n_samples, img_cond.device)
            # current prediction for x_0
            pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t.sqrt()
            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * torch.randn(size, device=img_cond.device) * 1.
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

            target_latent = LDModel.q_sample(target_im_z, t-1)
            return torch.clamp((x_prev + 1.0) / 2.0, min=0.0, max=1.0), target_latent
            # return (x_prev + 1.0) / 2.0, target_latent


def main_run(raw_im, target_im,
             models, device,
             gt_elevation=0.0, gt_azimuth=0.0, gt_radius=0.0,
             start_elevation=0.0, start_azimuth=0.0, start_radius=0.0,
             preprocess=True,
             scale=3.0, n_samples=1, ddim_steps=75, ddim_eta=1.0,
             precision='fp32', h=256, w=256,
             run_name=''):
    '''
    :param raw_im (PIL Image).
    '''
    # tb_writer = writer.SummaryWriter(f'../runs/')
    curr_time = time.localtime(time.time())
    yday = curr_time.tm_yday
    hours = curr_time.tm_hour
    mins = curr_time.tm_min + curr_time.tm_sec / 60
    if run_name:
        writer_name = f'../runs/{run_name}_gt-{np.rad2deg(gt_elevation):.0f}-{np.rad2deg(gt_azimuth):.0f}'+\
                      f'_st-{np.rad2deg(start_elevation):.0f}-{np.rad2deg(start_azimuth):.0f}'+\
                      f'_{yday}-{hours}-{mins:.1f}/'
    else:
        writer_name = f'../runs/gt-{np.rad2deg(gt_elevation):.0f}-{np.rad2deg(gt_azimuth):.0f}'+\
                      f'_st-{np.rad2deg(start_elevation):.0f}-{np.rad2deg(start_azimuth):.0f}'+\
                      f'_{yday}-{hours}-{mins:.1f}/'
    tb_writer = writer.SummaryWriter(writer_name)

    # print('ddim_steps=', ddim_steps)
    raw_im.thumbnail([1536, 1536], Image.Resampling.LANCZOS)
    target_im.thumbnail([1536, 1536], Image.Resampling.LANCZOS)
    """
    safety_checker_input = models['clip_fe'](raw_im, return_tensors='pt').to(device)
    (image, has_nsfw_concept) = models['nsfw'](
        images=np.ones((1, 3)), clip_input=safety_checker_input.pixel_values)
    print('has_nsfw_concept:', has_nsfw_concept)
    if np.any(has_nsfw_concept):
        print('NSFW content detected.')
        to_return = [None] * 10
        description = ('###  <span style="color:red"> Unfortunately, '
                       'potential NSFW content was detected, '
                       'which is not supported by our model. '
                       'Please try again with a different image. </span>')
        to_return[0] = description
        return to_return
    else:
        print('Safety check passed.')
    """

    input_im = preprocess_image(models, raw_im, preprocess, h=h, w=w, device=device)
    target_im = preprocess_image(models, target_im, preprocess, h=h, w=w, device=device)
    LDModel = models['turncam']
    LDModel.register_buffer('ddim_sigmas_for_original_steps', 
                            ddim_eta * torch.sqrt((1 - LDModel.alphas_cumprod_prev) / (1 - LDModel.alphas_cumprod) *
                                                  (1 - LDModel.alphas_cumprod / LDModel.alphas_cumprod_prev)))
    # sampler = DDIMSampler(models['turncam'])
    
    # used_x = -x  # NOTE: Polar makes more sense in Basile's opinion this way!
    # used_elevation = elevation  # NOTE: Set this way for consistency.
    start_elevation = Parameter(data=Tensor([start_elevation]).to(torch.float32).to(device), requires_grad=True)
    start_azimuth = Parameter(data=Tensor([start_azimuth]).to(torch.float32).to(device), requires_grad=True)
    start_radius = Parameter(data=Tensor([start_radius]).to(torch.float32).to(device), requires_grad=True)

    # optim = torch.optim.Adam([elevation, azimuth, radius], lr=1e-3), {'params': sampler.model.parameters()}
    # optimizer = optim.SGD([{'params': start_elevation},
    #                        {'params': start_azimuth},
    #                        {'params': start_radius, 'lr': 1e-10}], lr=0.2)
    optimizer = optim.Adam([{'params': [start_elevation, start_azimuth]},
                            {'params': start_radius, 'lr': 1e-10}], lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,50,100,150], gamma=0.5)

    max_iter = 2000
    from tqdm import tqdm
    iterator = tqdm(range(max_iter), desc='DDIM', total=max_iter)
    max_index = 50
    interval = max_iter/max_index
    step_interval = int(1000//75)
    for i, iter in enumerate(iterator):
        iterator.set_description_str(f'[{iter}/{max_iter}]')
        optimizer.zero_grad()
        
        index = int(1 + max_index - iter//interval)
        step = step_interval * index + 1
        x_samples_ddim, target_latent = sample_model(input_im, target_im, LDModel, precision, 
                                                     h, w, n_samples, scale, ddim_eta,
                                                     start_elevation, start_azimuth, start_radius, step)
        
        
        loss = torch.nn.functional.mse_loss(x_samples_ddim, target_latent, reduction='mean')
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            scheduler.step()
        with torch.no_grad():
            # region
            # while start_elevation > np.pi:
            #     start_elevation = start_elevation - 2*np.pi
            # while start_elevation < -np.pi:
            #     start_elevation = start_elevation + 2*np.pi
            # while start_azimuth < 0:
            #     start_azimuth = start_azimuth + 2*np.pi
            # while start_azimuth > 2*np.pi:
            #     start_azimuth = start_azimuth - 2*np.pi
            # endregion
            err = [np.rad2deg(start_elevation.item() - gt_elevation), np.rad2deg(start_azimuth.item() - gt_azimuth), start_radius.item() - gt_radius]
            temp_elev= np.rad2deg(start_elevation.item())
            temp_azi= np.rad2deg(start_azimuth.item())
            iterator.set_postfix_str(f'step: {index}-{step}, loss: {loss.item():.3f}, Err elev, azi= {err[0]:.2f}, {err[1]:.2f}, curr= {temp_elev:.2f}, {temp_azi:.2f}')
            tb_writer.add_scalar('Loss', loss.item(), iter)
            tb_writer.add_scalar('Error/elevation', err[0], iter)
            tb_writer.add_scalar('Error/azimuth', err[1], iter)
            tb_writer.add_scalar('Estimate/elevation', temp_elev, iter)
            tb_writer.add_scalar('Estimate/azimuth', temp_azi, iter)
    return 0

_GPU_INDEX = 0
def predict(device_idx: int = _GPU_INDEX,
            ckpt: str ="../105000.ckpt",
            config: str ="configs/sd-objaverse-finetune-c_concat-256.yaml",
            ref_image_path: str = "ref.png",
            target_image_path: str = "target.png",
            rel_elevation_in_degree: float = 0.0,
            rel_azimuth_in_degree: float = 0.0,
            rel_radius: float = 0.0,
            run_name: str = "test_run",
            ):
    device = f"cuda:{device_idx}"
    print('device = ',device)
    config_obj = OmegaConf.load(config)

    assert os.path.exists(ckpt)
    assert os.path.exists(ref_image_path)

    # Instantiate all models beforehand for efficiency.
    models = dict()
    print('Instantiating LatentDiffusion...')
    models['turncam'] = load_model_from_config(config_obj, ckpt, device=device, verbose=True)
    print('Instantiating Carvekit HiInterface...')
    models['carvekit'] = create_carvekit_interface()
    """
    print('Instantiating StableDiffusionSafetyChecker...')
    models['nsfw'] = StableDiffusionSafetyChecker.from_pretrained(
        'CompVis/stable-diffusion-safety-checker').to(device)
    """
    print('Instantiating AutoFeatureExtractor...')
    models['clip_fe'] = AutoFeatureExtractor.from_pretrained(
        'CompVis/stable-diffusion-safety-checker')

    ref_image = Image.open(ref_image_path)
    target_image = Image.open(target_image_path)
    gt_elev = 0
    gt_azi = 0
    match1 = re.search(r"elev=(-?[\d.]+)_azi=(-?[\d.]+)", ref_image_path)
    match2 = re.search(r"elev=(-?[\d.]+)_azi=(-?[\d.]+)", target_image_path)
    if match1 and match2:
        print(f"start_rel: elev= {rel_elevation_in_degree}, azi= {rel_azimuth_in_degree}")
        gt_elev = float(match2.group(1)) - float(match1.group(1))
        gt_azi = float(match2.group(2)) - float(match1.group(2))
        print(f"gt_rel: elev= {gt_elev}, azi= {gt_azi}")
    
    main_run(raw_im=ref_image, target_im=target_image,
                            models=models, device=device,
                            gt_elevation=np.deg2rad(gt_elev),
                            gt_azimuth=np.deg2rad(gt_azi),
                            gt_radius=0.0,
                            start_elevation=np.deg2rad(rel_elevation_in_degree),
                            start_azimuth=np.deg2rad(rel_azimuth_in_degree),
                            start_radius=rel_radius,
                            run_name=run_name)

if __name__ == '__main__':
    '''
    python test.py --ckpt "path_to_ckpt" \
        --ref_image_path "path_to_ref_image" \
        --target_image_path "path_to_target_image" \
        --rel_elevation_in_degree 0.0 \
        --rel_azimuth_in_degree 0.0 \
        --rel_radius 0.0
    '''
    '''
    python test.py --ckpt ../105000.ckpt \
        --ref_image_path "data/gso_alarm_my_render/elev=0_azi=0.png" \
        --target_image_path "data/gso_alarm_my_render/elev=0_azi=10.png" \
        --rel_elevation_in_degree 0.0 \
        --rel_azimuth_in_degree 0.0 \
        --rel_radius 0.0
        --run_name test
    '''
    # parser = argparse.ArgumentParser(**parser_kwargs)
    # parser.add_argument(
    #     "--finetune_from",
    #     type=str,
    #     nargs="?",
    #     default="",
    #     help="path to checkpoint to load model state from"
    # )
    fire.Fire(predict)
