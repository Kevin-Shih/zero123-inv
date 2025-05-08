from ast import Dict
from math import e
import os
# import math
# from cv2 import mean
from einops import rearrange
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
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from rich import print
from torch import Tensor, optim, nn, randint
from torch.nn.parameter import Parameter
from torch.amp.autocast_mode import autocast
from torch.utils.tensorboard import writer
from torchvision import transforms
from transformers import AutoFeatureExtractor

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
    # else:
    #     print(f'Infer foreground mask (preprocess_image) took {time.time() - start_time:.3f}s.')
    #     print('input_im:', lo(input_im))
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
            # region prepare input
            # Set time step and noisy latent shape
            t = torch.full((n_samples,), step, device=input_im.device, dtype=torch.long)
            size = (n_samples, 4, h // 8, w // 8)
            # Get input & target latent
            input_encoder_posterior = LDModel.encode_first_stage(input_im)
            input_im_z = LDModel.get_first_stage_encoding(input_encoder_posterior)
            target_encoder_posterior = LDModel.encode_first_stage(target_im)
            target_im_z = LDModel.get_first_stage_encoding(target_encoder_posterior)
            # Add noise to the input latent and target latent
            # _noise = torch.randn_like(input_im_z)
            _noise = torch.randn(size, device=input_im_z.device)
            input_latent = LDModel.q_sample(input_im_z, t, _noise)
            target_latent = LDModel.q_sample(target_im_z, t-1, _noise)
            # Get condintioning
            img_cond = LDModel.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = torch.cat([elevation, torch.sin(azimuth), torch.cos(azimuth), radius])
            T_batch = T[None, None, :].repeat(n_samples, 1, 1)
            c = torch.cat([img_cond, T_batch], dim=-1)
            c_proj = LDModel.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c_proj]
            cond['c_concat'] = [input_encoder_posterior.mode().repeat(n_samples, 1, 1, 1)]
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(size, device=img_cond.device)]
                uc['c_crossattn'] = [torch.zeros_like(c_proj, device=img_cond.device)]
            else:
                uc = None
            # endregion

            # print(f'conditioning_key= {LDModel.model.conditioning_key}')
            # print(f"target_im_z={target_im_z.shape}, target_latent={target_latent.shape}")
            if uc is None or scale == 1.:
                e_t = LDModel.apply_model(input_latent, t, cond)
            else:
                x_in = torch.cat([input_latent] * 2)
                t_in = torch.cat([t] * 2)
                if isinstance(cond, dict):
                    assert isinstance(uc, dict)
                    c_in = dict()
                    for k in cond:
                        if isinstance(cond[k], list):
                            c_in[k] = [torch.cat([uc[k][i],
                                       cond[k][i]]) for i in range(len(cond[k]))]
                        else:
                            c_in[k] = torch.cat([uc[k], cond[k]])
                else:
                    assert not isinstance(uc, dict)
                    c_in = torch.cat([uc, cond])
                e_t_uncond, e_t = LDModel.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + scale * (e_t - e_t_uncond)
            
            a_t, a_prev, sigma_t, sqrt_one_minus_at = calculate_param(LDModel, step, n_samples, img_cond.device)
            # current prediction for x_0
            pred_x0 = (input_latent - sqrt_one_minus_at * e_t) / a_t.sqrt()
            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + sigma_t * _noise

            return x_prev, target_latent, pred_x0, input_latent, target_im_z


def main_run(conf,
             raw_im, target_im,
             models, device,
             gt_elevation=0.0, gt_azimuth=0.0, gt_radius=0.0,
             start_elevation=0.0, start_azimuth=0.0, start_radius=0.0,
             preprocess=True,
             scale=3.0, n_samples=4, ddim_steps=75, ddim_eta=1.0,
             learning_rate = 1e-3,
             precision='fp32', h=256, w=256,
             tb_writer:writer.SummaryWriter=writer.SummaryWriter(),):
    '''
    :param raw_im (PIL Image).
    '''
    raw_im.thumbnail([1536, 1536], Image.Resampling.LANCZOS)
    target_im.thumbnail([1536, 1536], Image.Resampling.LANCZOS)

    input_im = preprocess_image(models, raw_im, preprocess, h=h, w=w, device=device)
    target_im = preprocess_image(models, target_im, preprocess, h=h, w=w, device=device)
    LDModel = models['turncam']
    LDModel.register_buffer('ddim_sigmas_for_original_steps', 
                            ddim_eta * torch.sqrt((1 - LDModel.alphas_cumprod_prev) / (1 - LDModel.alphas_cumprod) *
                                                  (1 - LDModel.alphas_cumprod / LDModel.alphas_cumprod_prev)))
    
    # used_x = -x  # NOTE: Polar makes more sense in Basile's opinion this way!
    # used_elevation = elevation  # NOTE: Set this way for consistency.
    start_elevation = Tensor([start_elevation]).to(torch.float32).to(device)
    start_radius = Tensor([start_radius]).to(torch.float32).to(device)
    latent_loss_all = []
    latent_x0_loss_all = []
    img_loss_all = []
    img_loss_x0_all = []
    decode_loss_all = []
    from tqdm import tqdm
    lower_bound = int(conf.check_range[0]*10)
    upper_bound = int(conf.check_range[1]*10)+1
    # print(f'lower_bound= {lower_bound}, upper_bound= {upper_bound}')
    azi_list = np.array([x for x in range(lower_bound, upper_bound, conf.check_step)]) / 10.0  + np.round(np.rad2deg(gt_azimuth),0) # all
    # azi_list = np.array([x for x in range(-1200, 1201, 5)]) / 10.0  + np.round(np.rad2deg(gt_azimuth),0) # wider
    # azi_list = np.array([x for x in range(-900, 901, 1)]) / 10.0  + np.round(np.rad2deg(gt_azimuth),0)   # wide
    # azi_list = np.array([x for x in range(-300, 301, 1)]) / 10.0  + np.round(np.rad2deg(gt_azimuth),0)   # normal
    # azi_list = np.array([x for x in range(-150, 151, 1)]) / 10.0  + np.round(np.rad2deg(gt_azimuth),0)   # narrow
    start_azimuth = Tensor(np.deg2rad(azi_list)).to(torch.float32).to(device)
    max_iter = len(azi_list)
    max_index = conf.input.max_index
    min_index = max_index if conf.input.min_index is None else max(conf.input.min_index, 0)
    for index in range(min_index, max_index+1):
        pbar = tqdm(range(max_iter), desc='DDIM', total=max_iter, ncols=140)
        
        decode_loss_index = []
        latent_loss_index = []
        latent_x0_loss_index = []
        img_loss_index = []
        img_loss_x0_index = []
        for i, iter in enumerate(pbar, start=1):
            pbar.set_description_str(f'[{i}/{max_iter}]')

            step = int(1000//75) * max(index, 0) + 1
            with torch.no_grad():
                pred_target, target_latent, pred_x0, input_latent, target_im_z = sample_model(input_im, target_im, LDModel, precision, 
                                                            h, w, n_samples, scale, ddim_eta,
                                                            start_elevation, start_azimuth[iter].unsqueeze(0), start_radius, step)
                decode_pred_target   = LDModel.decode_first_stage(pred_target)
                decode_input_latent  = LDModel.decode_first_stage(input_latent)
                decode_target_latent = LDModel.decode_first_stage(target_latent)

                decode_target_x0     = LDModel.decode_first_stage(target_im_z)
                decode_pred_x0       = LDModel.decode_first_stage(pred_x0)

                decode_pred_target   = torch.clamp((decode_pred_target   + 1.0) / 2.0, min=0.0, max=1.0)
                decode_input_latent  = torch.clamp((decode_input_latent  + 1.0) / 2.0, min=0.0, max=1.0)
                decode_target_latent = torch.clamp((decode_target_latent + 1.0) / 2.0, min=0.0, max=1.0)
                decode_target_x0     = torch.clamp((decode_target_x0     + 1.0) / 2.0, min=0.0, max=1.0)
                decode_pred_x0       = torch.clamp((decode_pred_x0       + 1.0) / 2.0, min=0.0, max=1.0)

                latent_loss    = nn.functional.mse_loss(pred_target, target_latent)
                latent_x0_loss = nn.functional.mse_loss(pred_x0, target_im_z.expand_as(pred_x0))
                # neg_latent_x0_loss = -1 * nn.functional.mse_loss(pred_x0, target_im_z.expand_as(pred_x0))
                img_loss       = nn.functional.mse_loss(decode_pred_target, decode_target_latent)
                img_x0_loss    = nn.functional.mse_loss(decode_pred_x0, target_im.expand_as(decode_pred_x0))
                decode_loss    = nn.functional.mse_loss(decode_target_x0, target_im.expand_as(decode_target_x0))
                
                latent_loss_index.append(latent_loss.item())
                latent_x0_loss_index.append(latent_x0_loss.item())
                img_loss_index.append(img_loss.item())
                img_loss_x0_index.append(img_x0_loss.item())
                decode_loss_index.append(decode_loss.item())

                total_loss = latent_loss + img_loss
                # region logging
            # if index == max_index:
                temp_elev= np.rad2deg(start_elevation.item())
                temp_azi= np.rad2deg(start_azimuth[iter].item())
                temp_radius= start_radius.item()

                err = [temp_elev - np.rad2deg(gt_elevation), temp_azi - np.rad2deg(gt_azimuth), temp_radius - gt_radius]
                pbar.set_postfix_str(f'step: {index}-{step}, loss: {total_loss.item():.3f}, Err elev, azi= {err[0]:.2f}, {err[1]:.2f}, Curr= {temp_elev:.2f}, {temp_azi:.2f}')
                tb_writer.add_scalar('Loss/total',      total_loss.item(),     iter)
                tb_writer.add_scalar('Loss/img',        img_loss.item(),       iter)
                tb_writer.add_scalar('Loss/latent',     latent_loss.item(),    iter)
                tb_writer.add_scalar('Loss/latent_x0',  latent_x0_loss.item(), iter)
                tb_writer.add_scalar('Loss/img_x0',     img_x0_loss.item(),    iter)
                tb_writer.add_scalar('Loss/decode',     decode_loss.item(),    iter)
                tb_writer.add_scalar('Error/elevation', err[0],         iter)
                tb_writer.add_scalar('Error/azimuth',   err[1],         iter)
                tb_writer.add_scalar('Error/Abs elev',  np.abs(err[0]), iter)
                tb_writer.add_scalar('Error/Abs azi',   np.abs(err[1]), iter)
                tb_writer.add_scalar('Estimate/elev',   temp_elev,      iter)
                tb_writer.add_scalar('Estimate/azi',    temp_azi,       iter)
                tb_writer.add_scalar('Log/ddim_index',  index,          iter)
                tb_writer.add_image('Image/generated_pred_target',   decode_pred_target[0],   iter)
                tb_writer.add_image('Image/generated_target_latent', decode_target_latent[0], iter)
                tb_writer.add_image('Image/generated_pred_x0',       decode_pred_x0[0],       iter)
                tb_writer.add_image('Image/generated_input_x0',      decode_input_latent[0],  iter)
                tb_writer.add_image('Image/generated_target_x0',     decode_target_x0[0],     iter)
                # endregion
        latent_loss_all.append(latent_loss_index)
        latent_x0_loss_all.append(latent_x0_loss_index)
        img_loss_all.append(img_loss_index)
        img_loss_x0_all.append(img_loss_x0_index)
        decode_loss_all.append(decode_loss_index)
    # region plotting
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,10))
    # fig, ax = plt.subplots(figsize=(10,10))
    x = np.array(azi_list) - int(np.rad2deg(gt_azimuth))
    ax.plot(x[::1], latent_loss_all[0][::1], 'ro-', label='latent_loss')
    ax.set_xlabel('Azimuth Error', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.legend(
        loc='best',
        fontsize=14,
        shadow=False,
        facecolor='#bbb',
        edgecolor='#000',
        title=f'Loss Curve @ index{max_index}',
        title_fontsize=14)
    # fig.savefig('../runs/imgs/lossCurve_latent_loss.png')
    tb_writer.add_figure('LossCurve/latent_pred_target', fig, 1)
    ax.cla()

    ax.plot(x[::1], latent_x0_loss_all[0][::1], 'mo-', label='latent_loss@t0')
    ax.set_xlabel('Azimuth Error', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.legend(
        loc='best',
        fontsize=14,
        shadow=False,
        facecolor='#bbb',
        edgecolor='#000',
        title=f'Loss Curve @ index{max_index}',
        title_fontsize=14)
    # fig.savefig('../runs/imgs/lossCurve_latent_t0_loss.png')
    tb_writer.add_figure('LossCurve/latent_t0', fig, 1)
    ax.cla()

    ax.plot(x[::1], img_loss_all[0][::1], 'bo-', label='img_loss')
    ax.set_xlabel('Azimuth Error', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.legend(
        loc='best',
        fontsize=14,
        shadow=False,
        facecolor='#bbb',
        edgecolor='#000',
        title=f'Loss Curve @ index{max_index}',
        title_fontsize=14)
    # fig.savefig('../runs/imgs/lossCurve_img_loss.png')
    tb_writer.add_figure('LossCurve/img_pred_target', fig, 1)
    ax.cla()

    ax.plot(x[::1], img_loss_x0_all[0][::1], 'co-', label='img_loss@t0')
    ax.set_xlabel('Azimuth Error', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.legend(
        loc='best',
        fontsize=14,
        shadow=False,
        facecolor='#bbb',
        edgecolor='#000',
        title=f'Loss Curve @ index{max_index}',
        title_fontsize=14)
    # fig.savefig('../runs/imgs/lossCurve_img_t0_loss.png')
    tb_writer.add_figure('LossCurve/img_t0', fig, 1)
    # a1 = ax.imshow(np.asarray(latent_loss_all), cmap='Reds', interpolation='none', extent=[-15.0, 15.0, 0, max_index])
    # ax.set_aspect('auto')
    # colorbar = fig.colorbar(a1, ax=ax)
    # fig.savefig('../runs/imgs/latent_loss.png')
    # tb_writer.add_figure('LossCurve/latent_pred_target', fig, 0)
    # colorbar.remove()

    # a2 = ax.imshow(np.asarray(img_loss_all), cmap='Reds', interpolation='none', extent=[-15.0, 15.0, 0, max_index])
    # ax.set_aspect('auto')
    # colorbar = fig.colorbar(a2, ax=ax)
    # fig.savefig('../runs/imgs/img_loss.png')
    # tb_writer.add_figure('LossCurve/img_pred_target', fig, 0)
    # colorbar.remove()

    # a3 = ax.imshow(np.asarray(latent_x0_loss_all), cmap='Reds', interpolation='none', extent=[-15.0, 15.0, 0, max_index])
    # ax.set_aspect('auto')
    # colorbar = fig.colorbar(a3, ax=ax)
    # fig.savefig('../runs/imgs/latent_t0_loss.png')
    # tb_writer.add_figure('LossCurve/latent_t0', fig, 0)
    # colorbar.remove()

    # a4 = ax.imshow(np.asarray(img_loss_x0_all), cmap='Reds', interpolation='none', extent=[-15.0, 15.0, 0, max_index])
    # ax.set_aspect('auto')
    # colorbar = fig.colorbar(a4, ax=ax)
    # fig.savefig('../runs/imgs/img_t0_loss.png')
    # tb_writer.add_figure('LossCurve/img_t0', fig, 0)
    # endregion
    tb_writer.flush()
    return 0
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/sanity_check.yaml",
        help="path to configs to load OmegaConf from"
    )
    args = parser.parse_args()
    print(f'Loading configs from {os.path.basename(args.config)}')
    conf = OmegaConf.load(args.config)
    print()
    print(OmegaConf.to_yaml(conf))
    
    ref_image_path = os.path.join(conf.dataroot, conf.input.ref_image)
    target_image_path = os.path.join(conf.dataroot, conf.input.target_image)
    rel_elev_deg = conf.input.rel_elev
    rel_azi_deg = conf.input.rel_azi
    rel_radius = conf.input.rel_radius
    device = f"cuda:{conf.model.gpu_idx}"
    model_config_obj = OmegaConf.load(conf.model.model_config)

    assert os.path.exists(conf.model.ckpt)
    assert os.path.exists(ref_image_path)

    # region Instantiate all models beforehand for efficiency.
    models = dict()
    print('Instantiating LatentDiffusion...')
    models['turncam'] = load_model_from_config(model_config_obj, conf.model.ckpt, device=device, verbose=True)
    print('Instantiating Carvekit HiInterface...')
    models['carvekit'] = create_carvekit_interface()
    print('Instantiating AutoFeatureExtractor...')
    models['clip_fe'] = AutoFeatureExtractor.from_pretrained(
        'CompVis/stable-diffusion-safety-checker')
    # endregion

    # region Load images and set gt
    ref_image = Image.open(ref_image_path)
    target_image = Image.open(target_image_path)
    gt_elev = 0
    gt_azi = 0
    match1 = re.search(r"elev=(-?[\d.]+)_azi=(-?[\d.]+)", ref_image_path)
    match2 = re.search(r"elev=(-?[\d.]+)_azi=(-?[\d.]+)", target_image_path)
    if match1 and match2:
        print(f"start_rel: elev= {rel_elev_deg:5.1f}, azi= {rel_azi_deg:5.1f}")
        gt_elev = float(match2.group(1)) - float(match1.group(1))
        gt_azi = float(match2.group(2)) - float(match1.group(2))
        if gt_azi > 180:
            gt_azi -= 360
        print(f"gt_rel:    elev= {gt_elev:5.1f}, azi= {gt_azi:5.1f}")
    # endregion

    # region tb_writer setup
    curr_time = time.localtime(time.time())
    mon = curr_time.tm_mon
    mday = curr_time.tm_mday
    hours = curr_time.tm_hour
    mins = curr_time.tm_min + curr_time.tm_sec / 60
    if conf.log.run_name != '':
        writer_name = f'{conf.log.log_root}/{mon:02d}-{mday:02d}/{hours}-{mins:.1f}_{conf.log.run_name}_gt-{gt_elev:.0f}-{gt_azi:.0f}'+\
                      f'_st-{rel_elev_deg:.0f}-{rel_azi_deg:.0f}'
    else:
        writer_name = f'{conf.log.log_root}/{mon:02d}-{mday:02d}/{hours}-{mins:.1f}_gt-{gt_elev:.0f}-{gt_azi:.0f}'+\
                      f'_st-{rel_elev_deg:.0f}-{rel_azi_deg:.0f}'
    tb_writer = writer.SummaryWriter(writer_name)
    # endregion
    
    main_run(conf = conf,
             raw_im = ref_image, target_im = target_image,
             models = models, device = device,
             learning_rate = conf.model.lr,
             gt_elevation = np.deg2rad(gt_elev),
             gt_azimuth = np.deg2rad(gt_azi),
             gt_radius = 0.0,
             start_elevation = np.deg2rad(rel_elev_deg),
             start_azimuth = np.deg2rad(rel_azi_deg),
             start_radius = conf.input.rel_radius,
             tb_writer = tb_writer,
             n_samples= conf.model.n_samples
             )