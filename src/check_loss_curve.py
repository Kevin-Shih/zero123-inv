import os
import argparse
import numpy as np
import time
import re
import wandb
import yaml

import torch
from tqdm import tqdm
from contextlib import nullcontext
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import create_carvekit_interface, load_and_preprocess, instantiate_from_config
from lovely_numpy import lo
from omegaconf import OmegaConf
from PIL import Image
from rich import print
from torch import Tensor, optim, nn
from torch.amp.autocast_mode import autocast
from torchvision import transforms
from transformers import AutoFeatureExtractor
from einops import rearrange


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
    # model.train()
    model.eval()
    return model

def preprocess_image(models, input_im, preprocess, h=256, w=256, device='cuda'):
    '''
    :param input_im (PIL Image).
    :return input_im (H, W, 3) array in [0, 1].
    '''
    old_size = input_im.size
    start_time = time.time()

    if preprocess:
        input_im, forground_mask = load_and_preprocess(models['carvekit'], input_im)
        input_im = (input_im / 255.0).astype(np.float32)
        # forground_mask = forground_mask.astype(np.float32)
        # (H, W, 3) array in [0, 1].
    else:
        input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0
        # (H, W, 4) array in [0, 1].

        # old method: thresholding background, very important
        # input_im[input_im[:, :, -1] <= 0.9] = [1., 1., 1., 1.]

        # new method: apply correct method of compositing to avoid sudden transitions / thresholding
        # (smoothly transition foreground to white background based on alpha values)
        forground_mask = np.zeros([256, 256], dtype=np.float32) 
        forground_mask[input_im[:, :, -1] > 0.9] = [1.]
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
    input_im = input_im * 2 - 1 # move to [-1, 1]
    input_im = transforms.Resize([h, w])(input_im)
    return input_im, forground_mask

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

def calculate_param_ddim(sampler, index, n_samples, device):
    alphas = sampler.ddim_alphas
    alphas_prev = sampler.ddim_alphas_prev
    sigmas = sampler.ddim_sigmas
    sqrt_one_minus_alphas = sampler.ddim_sqrt_one_minus_alphas
    # print(index, alphas.shape)
    # select parameters corresponding to the currently considered timestep
    a_t = torch.full((n_samples, 1, 1, 1), alphas[index], device=device)
    a_prev = torch.full((n_samples, 1, 1, 1), alphas_prev[index], device=device)
    sigma_t = torch.full((n_samples, 1, 1, 1), sigmas[index], device=device)
    sqrt_one_minus_at = torch.full((n_samples, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)    
    return a_t, a_prev, sigma_t, sqrt_one_minus_at

def sample_model(input_im, target_im, LDModel, precision, h, w,
                 elevation, azimuth, radius, n_samples, 
                 scale = 3.0, ddim_steps= 75, ddim_eta= 0.15, index = 5):
    sampler = DDIMSampler(LDModel)
    sampler.make_schedule(ddim_num_steps=75, ddim_discretize="uniform", ddim_eta=ddim_eta, verbose=False)
    # if index is not None:
    #     step = np.flip(sampler.ddim_timesteps)[index]
    step = int(1000//ddim_steps) * max(index, 0) + 1
    step_target_inter = int(1000//ddim_steps) if index > 0 else 1
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with LDModel.ema_scope():
            # region prepare input
            # Set time step and noisy latent shape
            t = torch.full((n_samples,), step, device=input_im.device, dtype=torch.long) # type: ignore
            size = (n_samples, 4, h // 8, w // 8)
            # Get input & target latent
            input_encoder_posterior = LDModel.encode_first_stage(input_im)
            input_im_z = LDModel.get_first_stage_encoding(input_encoder_posterior).detach()
            # input_im_z = input_encoder_posterior.mode().detach()
            target_encoder_posterior = LDModel.encode_first_stage(target_im)
            target_im_z = LDModel.get_first_stage_encoding(target_encoder_posterior).detach()
            # target_im_z = target_encoder_posterior.mode().detach()
            # Add noise to the input latent and target latent
            # _noise = torch.randn_like(input_im_z)
            _noise = torch.randn(size, device=input_im_z.device)
            # _perfect_input = target_im_z.clone().detach()
            input_latent = LDModel.q_sample(target_im_z.clone().detach(), t, _noise)
            # input_latent = LDModel.q_sample(input_im_z, t, _noise)
            _target_start_latent = LDModel.q_sample(target_im_z.clone().detach(), t, _noise)
            target_latent = LDModel.q_sample(target_im_z, t-step_target_inter, _noise)

            latent_diff = _target_start_latent - target_latent
            latent_x0_diff = _target_start_latent - target_im_z
            # Get condintioning
            img_cond = LDModel.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = torch.cat([elevation, torch.sin(azimuth), torch.cos(azimuth), radius])
            T_batch = T[None, None, :].repeat(n_samples, 1, 1)
            c = torch.cat([img_cond, T_batch], dim=-1)
            c_proj = LDModel.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c_proj]
            # cond['c_concat'] = [input_im_z.clone().detach().repeat(n_samples, 1, 1, 1)]
            cond['c_concat'] = [input_encoder_posterior.mode().detach().repeat(n_samples, 1, 1, 1)]
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
            
            # a_t, a_prev, sigma_t, sqrt_one_minus_at = calculate_param(LDModel, step, n_samples, img_cond.device)
            a_t, a_prev, sigma_t, sqrt_one_minus_at = calculate_param_ddim(sampler, index, n_samples, img_cond.device)
            # current prediction for x_0
            pred_x0 = (input_latent - sqrt_one_minus_at * e_t) / a_t.sqrt()
            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            pred_x_idx_minus_one = a_prev.sqrt() * pred_x0 + dir_xt + sigma_t * _noise
            noise_loss = torch.nn.functional.mse_loss(LDModel.apply_model(input_latent, t, cond), _noise)

            return pred_x_idx_minus_one, pred_x0, input_latent, target_latent, target_im_z, latent_diff, latent_x0_diff, noise_loss

class PoseT(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pose):

        p1 = pose[..., 0:1]
        p2 = torch.sin(pose[..., 1:2])
        p3 = torch.cos(pose[..., 1:2])
        p4 = pose[..., 2:]

        return torch.cat([p1, p2, p3, p4], dim=-1)

def create_pose_params(pose, device):

    theta = torch.tensor([pose[0]], requires_grad=True, device=device)
    azimuth = torch.tensor([pose[1]], requires_grad=True, device=device)
    radius = torch.tensor([pose[2]], requires_grad=True, device=device)

    return [theta, azimuth, radius]

def create_random_pose():

    theta = np.random.rand() * np.pi - np.pi / 2
    azimuth = np.random.rand() * np.pi * 2
    radius = np.random.rand() - 0.5
    
    return [theta, azimuth, radius]

def get_inv_pose(pose):

    return [-pose[0], np.pi*2 - pose[1], -pose[2]]

def find_optimal_poses(model, images, learning_rate, bsz=1, n_iter=1000, init_poses={}, ts_range=[0.02, 0.92], combinations=None, print_n=50, avg_last_n=1):
    
    layer = PoseT()

    num = len(images)

    batch = {}

    pose_params = { i:[] for i in range(1, num)}
    pose_trajs = { i:[]  for i in range(1, num) }

    for i in range(1, num):

        if i in init_poses:
            init_pose = init_poses[i]
        else:
            init_pose = create_random_pose()

        pose = create_pose_params(init_pose, model.device)
        pose_params[i] = pose

    if combinations is None:
        combinations = []
        for i in range(0, num):
            for j in range(i+1, num):
                combinations.append((i, j))
                combinations.append((j, i))

    param_list = []
    for i in pose_params:
        param_list += pose_params[i]

    optimizer = torch.optim.SGD(param_list, lr = learning_rate)

    loss_traj = []
    select_indces = set([])
                                
    for iter in range(0, n_iter):

        if print_n > 0 and iter % print_n == 0 and iter > 0:
            print(iter, np.mean(loss_traj[-avg_last_n:]), flush=True)
            for i in range(1, num):
                print(0, i, np.mean(pose_trajs[i][-avg_last_n:], axis=0).tolist())

        '''record poses'''
        for i in select_indces:
            pose = pose_params[i]
            pose_trajs[i].append([pose[0].item(), pose[1].item(), pose[2].item()])

        select_indces = set([])

        conds = []
        targets = []
        rts = []

        choices = [ iter % len(combinations) ] 
        
        if bsz > 1:
            choices = np.random.choice(len(combinations), size=bsz, replace=True)

        for cho in choices:

            i, j = combinations[cho]

            conds.append(images[i])
            targets.append(images[j])
            if i == 0:
                pose = pose_params[j]
                select_indces.add(j)
            
            elif j == 0:
                pose = get_inv_pose(pose_params[i])
                select_indces.add(i)

            else:
                pose0j = pose_params[j]
                posei0 = get_inv_pose(pose_params[i])

                if np.random.rand() < 0.5:
                    posei0 = [a.item() for a in posei0]
                    select_indces.add(j)
                else:
                    pose0j = [b.item() for b in pose0j]
                    select_indces.add(i)

                #pose = [ torch.remainder(a+b+2*np.pi, 2*np.pi) - np.pi for a, b in zip(posei0, pose0j) ]
                pose = [ a+b for a, b in zip(posei0, pose0j) ]

            rts.append(torch.cat(pose)[None, ...])

        batch['image_cond'] = torch.cat(conds, dim=0)
        batch['image_target'] = torch.cat(targets, dim=0)
        batch['T'] = layer(torch.cat(rts, dim=0))
        ts = np.arange(ts_range[0], ts_range[1], (ts_range[1]-ts_range[0]) / len(conds))
        # print('ts=',ts)
        # print(f'batch["image_cond"].shape: {batch["image_cond"].shape}')
        # print(f'conds.shape: {len(conds), conds[0].shape}')
        optimizer.zero_grad()
        loss, loss_dict = model.shared_step(batch, ts=ts)
        loss.backward()

        optimizer.step()

        loss_traj.append(loss.item())

    if n_iter > 0:
        result_poses = [np.mean(pose_trajs[i][-avg_last_n:], axis=0).tolist() for i in range(1, num) ]
        result_loss = np.mean(loss_traj[-avg_last_n:])
    else:
        result_poses = [ init_poses[i] for i in range(1, num) ]
        result_loss = None

    return result_poses, [ init_poses[i] for i in range(1, num) ], result_loss

def idp_noise_loss(model, cond_image, target_image, pose, ts_range, bsz, noise=None):

    mx = ts_range[1]
    mn = ts_range[0]

    pose_layer = PoseT()

    batch = {}
    batch['image_target'] = target_image.repeat(bsz, 1, 1, 1)
    batch['image_cond'] = cond_image.repeat(bsz, 1, 1, 1)
    batch['T'] = pose_layer(pose.detach()).repeat(bsz, 1)

    if noise is not None:
        noise = torch.tensor(noise, dtype=model.dtype, device=model.device)
    else:
        noise = torch.randn(bsz, 4, 32, 32, device=model.device)
    loss, loss_dict = model.shared_step(batch, ts=np.arange(mn, mx, (mx-mn) / bsz), noise=noise[:bsz])

    return loss.item(), loss_dict


def idp_pairwise_loss(pose, model, cond_image, target_image, ts_range, probe_bsz, noise=None):

    theta, azimuth, radius = pose

    pose1 = torch.tensor([[theta, azimuth, radius]], device=model.device, dtype=torch.float32)
    pose2 = torch.tensor([[-theta, np.pi*2-azimuth, -radius]], device=model.device, dtype=torch.float32)
    loss1, loss_dict1 = idp_noise_loss(model, cond_image, target_image, pose1, ts_range, probe_bsz, noise=noise)
    loss2, loss_dict2 = idp_noise_loss(model, target_image, cond_image, pose2, ts_range, probe_bsz, noise=noise)

    return loss1 + loss2, loss_dict1, loss_dict2

def main_run(conf,
             input_im, target_im,
             models, device,
             gt_elevation=0.0, gt_azimuth=0.0, gt_radius=0.0,
             start_elevation=0.0, start_azimuth=0.0, start_radius=0.0,
             preprocess=True,
             scale=3.0, n_samples=4, ddim_steps=75, ddim_eta=1.0,
             learning_rate = 1e-3,
             precision='fp32', h=256, w=256,
             ):
    '''
    :param raw_im (PIL Image).
    '''
    input_im.thumbnail([256, 256], Image.Resampling.LANCZOS)
    target_im.thumbnail([256, 256], Image.Resampling.LANCZOS)

    input_im, _ = preprocess_image(models, input_im, preprocess, h=h, w=w, device=device)
    target_im, target_mask = preprocess_image(models, target_im, preprocess, h=h, w=w, device=device)
    target_mask = Tensor(target_mask).to(device)

    _target_im_copy = target_im.clone().detach()
    if conf.model.clamp is not None:
        # print(f'clamping output: {conf.model.clamp}')
        if conf.model.clamp == 'normal':
            _target_im_copy      = torch.clamp((_target_im_copy      + 1.0) / 2.0, min=0.0, max=1.0)
        elif conf.model.clamp == 'ddim':
            _target_im_copy      = torch.clamp(_target_im_copy     , min=-1.0, max=1.0)
        elif "partial" in conf.model.clamp:
            pass

    clamp_input_im = torch.clamp((input_im + 1.0) / 2.0, min=0.0, max=1.0).cpu()
    clamp_target_im = torch.clamp((target_im + 1.0) / 2.0, min=0.0, max=1.0).cpu()

    LDModel = models['turncam']
    # LDModel.register_buffer('ddim_sigmas_for_original_steps', 
    #                         ddim_eta * torch.sqrt((1 - LDModel.alphas_cumprod_prev) / (1 - LDModel.alphas_cumprod) *
    #                                               (1 - LDModel.alphas_cumprod / LDModel.alphas_cumprod_prev)))

    # used_x = -x  # NOTE: Polar makes more sense in Basile's opinion this way!
    # used_elevation = elevation  # NOTE: Set this way for consistency.
    start_elevation = Tensor([start_elevation]).to(torch.float32).to(device)
    start_radius = Tensor([start_radius]).to(torch.float32).to(device)
    latent_loss_all = []
    latent_x0_loss_all = []
    img_loss_all = []
    img_loss_x0_all = []
    decode_loss_all = []
    
    lower_bound = int(conf.check_range[0]*10)
    upper_bound = int(conf.check_range[1]*10)+1
    # print(f'lower_bound= {lower_bound}, upper_bound= {upper_bound}')
    azi_list = np.array([x for x in range(lower_bound, upper_bound, conf.check_step)]) / 10.0  + np.round(np.rad2deg(gt_azimuth),0) # all
    # azi_list = np.array([x for x in range(-1200, 1201, 5)]) / 10.0  + np.round(np.rad2deg(gt_azimuth),0) # wider
    # azi_list = np.array([x for x in range(-900, 901, 1)]) / 10.0  + np.round(np.rad2deg(gt_azimuth),0)   # wide
    # azi_list = np.array([x for x in range(-450, 451, 1)]) / 10.0  + np.round(np.rad2deg(gt_azimuth),0)   # normal
    start_azimuth = Tensor(np.deg2rad(azi_list)).to(torch.float32).to(device)
    max_iter = len(azi_list)
    max_index = conf.input.max_index
    min_index = max_index if conf.input.min_index is None else max(conf.input.min_index, 0)
    curr_time = time.localtime(time.time())
    group_mon = curr_time.tm_mon
    group_mday = curr_time.tm_mday
    group_hours = curr_time.tm_hour
    group_mins = curr_time.tm_min // 5 * 5
    blur = transforms.GaussianBlur(kernel_size=[conf.model.blur_k_size], sigma = (conf.model.blur_min, conf.model.blur_max))
    for i, index in enumerate(range(min_index, max_index + 1, 2), 0):
        decode_loss_index = []
        latent_loss_index = []
        latent_x0_loss_index = []
        img_loss_index = []
        img_loss_x0_index = []
        # region summary setup
        # maybe use yaml to dict. directly put config file into w&b config.
        curr_time = time.localtime(time.time())
        mon = curr_time.tm_mon
        mday = curr_time.tm_mday
        hours = curr_time.tm_hour
        mins = curr_time.tm_min + curr_time.tm_sec / 60
        # writer_name = f'{mon:02d}-{mday:02d}/{hours}-{mins:.1f}_{conf.run_name}_gt-{gt_elev_deg:.0f}-{gt_azi_deg:.0f}'+\
        #               f'_st-{rel_elev_deg:.0f}-{rel_azi_deg:.0f}'
        
        wb_run = wandb.init(
            entity="kevin-shih",
            project="Zero123-Adv-Loss-Check",
            group= f'{conf.group_name}_{group_mon:02d}-{group_mday:02d}_{group_hours:02d}_{group_mins:02d}' if max_index > min_index else conf.group_name,
            name= f'{conf.run_name}_gt-{gt_elev_deg:.0f}-{gt_azi_deg:.0f}_idx{index}',
            settings=wandb.Settings(x_disable_stats=True),
            config={
                "start_date": f'{mon:02d}-{mday:02d}',
                "start_time": f'{hours:02d}-{mins:4.1f}',
                "gt_elev_deg": f'{gt_elev_deg:.1f}',
                "gt_azi_deg": f'{gt_azi_deg:.1f}',
                **yaml.load(open(args.config, 'r'), Loader=yaml.SafeLoader),
            },
        )
        # endregion
        wb_run.log({'Image/input_im':  wandb.Image(clamp_input_im[0],  caption=f"input_im"),
                    'Image/target_im': wandb.Image(clamp_target_im[0], caption=f"target_im"),
        }, step=0)

        pbar = tqdm(range(max_iter), desc='DDIM', total=max_iter, ncols=140)
        for j, iter in enumerate(pbar, start=1):
            pbar.set_description_str(f'[{j}/{max_iter}]')
            with torch.no_grad():
                pred_target, pred_x0, input_latent, target_latent, target_im_z,\
                latent_diff, latent_x0_diff, noise_loss = sample_model(input_im, target_im, LDModel, precision, h, w, 
                                                           start_elevation, start_azimuth[iter].unsqueeze(0), 
                                                           start_radius, n_samples= n_samples, scale= scale,
                                                           ddim_steps= ddim_steps, ddim_eta= ddim_eta, index= index)
                decode_pred_target   = LDModel.decode_first_stage(pred_target)
                decode_input_latent  = LDModel.decode_first_stage(input_latent)
                decode_target_latent = LDModel.decode_first_stage(target_latent)

                decode_target_x0     = LDModel.decode_first_stage(target_im_z)
                decode_pred_x0       = LDModel.decode_first_stage(pred_x0)
                
                if conf.model.clamp is not None:
                    # print(f'clamping output: {conf.model.clamp}')
                    if conf.model.clamp == 'normal' or conf.model.clamp == 'partial_normal':
                        decode_pred_target   = torch.clamp((decode_pred_target   + 1.0) / 2.0, min=0.0, max=1.0)
                        decode_input_latent  = torch.clamp((decode_input_latent  + 1.0) / 2.0, min=0.0, max=1.0)
                        decode_target_latent = torch.clamp((decode_target_latent + 1.0) / 2.0, min=0.0, max=1.0)
                        decode_target_x0     = torch.clamp((decode_target_x0     + 1.0) / 2.0, min=0.0, max=1.0)
                        decode_pred_x0       = torch.clamp((decode_pred_x0       + 1.0) / 2.0, min=0.0, max=1.0)
                    elif conf.model.clamp == 'ddim':
                        decode_pred_target   = torch.clamp(decode_pred_target  , min=-1.0, max=1.0)
                        decode_input_latent  = torch.clamp(decode_input_latent , min=-1.0, max=1.0)
                        decode_target_latent = torch.clamp(decode_target_latent, min=-1.0, max=1.0)
                        decode_target_x0     = torch.clamp(decode_target_x0    , min=-1.0, max=1.0)
                        decode_pred_x0       = torch.clamp(decode_pred_x0      , min=-1.0, max=1.0)
                
                blur_decode_pred_target     = blur(decode_pred_target)
                blur_decode_pred_x0         = blur(decode_pred_x0)
                blur_decode_target_latent   = blur(decode_target_latent)
                blur_target_im              = blur(_target_im_copy)
                blur_decode_target_x0       = blur(decode_target_x0)

                mask_blur_decode_pred_target     = blur_decode_pred_target   * target_mask.float()
                mask_blur_decode_pred_x0         = blur_decode_pred_x0       * target_mask.float()
                mask_blur_decode_target_latent   = blur_decode_target_latent * target_mask.float()
                mask_blur_target_im              = blur_target_im * target_mask.float()
                
                pred_latent_diff    = input_latent - pred_target
                pred_latent_x0_diff = input_latent - pred_x0

                # decode_loss       = nn.functional.mse_loss(decode_target_x0, _target_im_copy.expand_as(decode_target_x0))
                blur_decode_loss  = nn.functional.mse_loss(blur_decode_target_x0, blur_target_im.expand_as(decode_target_x0))
                input_latent_loss = nn.functional.mse_loss(input_latent, target_latent)
                latent_diff_loss        = nn.functional.mse_loss(pred_latent_diff, latent_diff) # loss between latent_diff and denoised diff
                latent_diff_x0_loss     = nn.functional.mse_loss(pred_latent_x0_diff, latent_x0_diff) # loss between latent_x0_diff and denoised x0 diff
                latent_loss       = nn.functional.mse_loss(pred_target, target_latent)
                latent_x0_loss    = nn.functional.mse_loss(pred_x0, target_im_z.expand_as(pred_x0))
                # img_loss          = nn.functional.mse_loss(decode_pred_target, decode_target_latent)
                # img_x0_loss       = nn.functional.mse_loss(decode_pred_x0, _target_im_copy.expand_as(decode_pred_x0))
                blur_img_loss     = nn.functional.mse_loss(blur_decode_pred_target, blur_decode_target_latent)
                blur_img_x0_loss  = nn.functional.mse_loss(blur_decode_pred_x0, blur_target_im.expand_as(blur_decode_pred_x0))
                # img_loss          = nn.functional.mse_loss(decode_pred_target, decode_target_latent) - decode_loss
                # img_x0_loss       = nn.functional.mse_loss(decode_pred_x0, _target_im_copy.expand_as(decode_pred_x0)) - decode_loss
                # blur_img_loss     = nn.functional.mse_loss(blur_decode_pred_target, blur_decode_target_latent) - blur_decode_loss
                # blur_img_x0_loss  = nn.functional.mse_loss(blur_decode_pred_x0, blur_target_im.expand_as(blur_decode_pred_x0)) - blur_decode_loss
                
                no_reduct_mse       = nn.MSELoss(reduction='none')
                non_zero_elements   = target_mask.sum()
                _mask_img_loss      = (no_reduct_mse(decode_pred_target, decode_target_latent) * target_mask.float()).sum()
                mask_img_loss       = _mask_img_loss / non_zero_elements
                _mask_img_x0_loss   = (no_reduct_mse(decode_pred_x0, _target_im_copy.expand_as(decode_pred_x0)) * target_mask.float()).sum()
                mask_img_x0_loss    = _mask_img_x0_loss / non_zero_elements
                
                _blur_mask_img_loss     = (no_reduct_mse(blur_decode_pred_target, blur_decode_target_latent) * target_mask.float()).sum()
                blur_mask_img_loss      = _blur_mask_img_loss / non_zero_elements
                _blur_mask_img_x0_loss  = (no_reduct_mse(blur_decode_pred_x0, blur_target_im.expand_as(blur_decode_pred_x0)) * target_mask.float()).sum()
                blur_mask_img_x0_loss   = _blur_mask_img_x0_loss / non_zero_elements

                # region alternate loss
                # _mask_img_loss      = (no_reduct_mse(decode_pred_target, decode_target_latent) * target_mask.float()).sum()
                # mask_img_loss       = _mask_img_loss / non_zero_elements - decode_loss
                # _mask_img_x0_loss   = (no_reduct_mse(decode_pred_x0, _target_im_copy.expand_as(decode_pred_x0)) * target_mask.float()).sum()
                # mask_img_x0_loss    = _mask_img_x0_loss / non_zero_elements - decode_loss
                
                # _blur_mask_img_loss     = (no_reduct_mse(blur_decode_pred_target, blur_decode_target_latent) * target_mask.float()).sum()
                # blur_mask_img_loss      = _blur_mask_img_loss / non_zero_elements - blur_decode_loss
                # _blur_mask_img_x0_loss  = (no_reduct_mse(blur_decode_pred_x0, blur_target_im.expand_as(blur_decode_pred_x0)) * target_mask.float()).sum()
                # blur_mask_img_x0_loss   = _blur_mask_img_x0_loss / non_zero_elements  - blur_decode_loss
                
                # latent_loss_index.append(latent_loss.item())
                # latent_x0_loss_index.append(latent_x0_loss.item())
                # img_loss_index.append(img_loss.item())
                # img_loss_x0_index.append(img_x0_loss.item())
                # decode_loss_index.append(decode_loss.item())
                #endregion

                idp_pose=torch.cat([start_elevation, start_azimuth[iter].unsqueeze(0), start_radius], dim=-1)
                raw_input_im = rearrange(input_im,'b c h w -> b h w c')
                raw_target_im = rearrange(target_im,'b c h w -> b h w c')
                idp_single_loss, loss_dict = idp_noise_loss(LDModel, raw_input_im, raw_target_im, idp_pose,
                                                ts_range=[0.2, 0.21], bsz=1, noise=None)
                idp_pair_loss,_,_ = idp_pairwise_loss(idp_pose, LDModel, raw_input_im, raw_target_im, ts_range=[0.2, 0.21], probe_bsz=1, noise=None)


                total_loss = blur_mask_img_x0_loss.clone().detach()
                if conf.model.loss_amp:
                    total_loss = total_loss * conf.model.loss_amp
            # region logging
                temp_elev= np.rad2deg(start_elevation.item())
                temp_azi= np.rad2deg(start_azimuth[iter].item())
                temp_radius= start_radius.item()

                err = [temp_elev - np.rad2deg(gt_elevation), temp_azi - np.rad2deg(gt_azimuth), temp_radius - gt_radius]
                pbar.set_postfix_str(f'step: {index}-{int(1000//ddim_steps) * max(index, 0) + 1}, ' +
                                     f'loss: {total_loss.item():.3f}, Err elev, azi= {err[0]:.2f}, ' + 
                                     f'{err[1]:.2f}, Curr= {temp_elev:.2f}, {temp_azi:.2f}')
                wb_run.log({
                    'Log/ddim_index':           index,
                    "Estimate/elevation":       temp_elev,
                    "Estimate/azimuth":         temp_azi,
                    'Error/elevation':          err[0],
                    'Error/azimuth':            err[1],
                    # 'Error/Abs elevation':      np.abs(err[0]),
                    # 'Error/Abs azimuth':        np.abs(err[1]),
                    # 'Loss/img':                 img_loss.item(),
                    # 'Loss/img_x0':              img_x0_loss.item(),
                    'Loss/mask_img':            mask_img_loss.item(),
                    'Loss/mask_img_x0':         mask_img_x0_loss.item(),
                    'Loss/blur_img':            blur_img_loss.item(),
                    'Loss/blur_img_x0':         blur_img_x0_loss.item(),
                    'Loss/blur_mask_img':       blur_mask_img_loss.item(),
                    'Loss/blur_mask_img_x0':    blur_mask_img_x0_loss.item(),
                    'Loss/latent_diff_loss':          latent_diff_loss.item(),
                    'Loss/latent_diff_x0_loss':       latent_diff_x0_loss.item(),
                    'Loss/latent':              latent_loss.item(),
                    'Loss/latent_x0':           latent_x0_loss.item(),
                    'Loss/neg_latent_x0':       -1 * latent_x0_loss.item(),
                    'Loss/input_latent':        input_latent_loss.item(),
                    # 'Loss/decode':              decode_loss.item(),
                    'Loss/blur_decode':         blur_decode_loss.item(),
                    'Loss/idp_single_loss':     idp_single_loss,
                    'Loss/idp_pair_loss':       idp_pair_loss,
                    'Loss/noise_loss':          noise_loss.item(),
                    'Loss/idp_simple_loss':     loss_dict['val/loss_simple'].item(),
                    'Loss/idp_vlb_loss':        loss_dict['val/loss_vlb'].item(),
                }, step=iter)
                if iter % 10 == 0:
                    wb_run.log({
                        'Image/input_latent':               wandb.Image(torch.clamp((decode_input_latent[0]            + 1.0) / 2.0, min=0.0, max=1.0), caption=f"generated_input_latent"),
                        'Image/target_x0':                  wandb.Image(torch.clamp((decode_target_x0[0]               + 1.0) / 2.0, min=0.0, max=1.0), caption=f"generated_target_x0"),
                        'Image/blured_target_im':           wandb.Image(torch.clamp((blur_target_im[0]                 + 1.0) / 2.0, min=0.0, max=1.0), caption=f"blured_target_im"),
                        'Image/blured_mask_target_im':      wandb.Image(torch.clamp((mask_blur_target_im[0]            + 1.0) / 2.0, min=0.0, max=1.0), caption=f"blur_mask_target_im"),
                        'Image/pred_target':                wandb.Image(torch.clamp((decode_pred_target[0]             + 1.0) / 2.0, min=0.0, max=1.0), caption=f"generated_pred_target"),
                        'Image/target_latent':              wandb.Image(torch.clamp((decode_target_latent[0]           + 1.0) / 2.0, min=0.0, max=1.0), caption=f"generated_target_latent"),
                        'Image/pred_x0':                    wandb.Image(torch.clamp((decode_pred_x0[0]                 + 1.0) / 2.0, min=0.0, max=1.0), caption=f"generated_pred_x0"),
                        'Image/blured_pred_target':         wandb.Image(torch.clamp((blur_decode_pred_target[0]        + 1.0) / 2.0, min=0.0, max=1.0), caption=f"blured_pred_target"),
                        'Image/blured_target_latent':       wandb.Image(torch.clamp((blur_decode_target_latent[0]      + 1.0) / 2.0, min=0.0, max=1.0), caption=f"blured_target_latent"),
                        'Image/blured_pred_x0':             wandb.Image(torch.clamp((blur_decode_pred_x0[0]            + 1.0) / 2.0, min=0.0, max=1.0), caption=f"blured_pred_x0"),
                        'Image/blured_mask_pred_target':    wandb.Image(torch.clamp((mask_blur_decode_pred_target[0]   + 1.0) / 2.0, min=0.0, max=1.0), caption=f"blur_mask_decode_pred_target"),
                        'Image/blured_mask_target_latent':  wandb.Image(torch.clamp((mask_blur_decode_target_latent[0] + 1.0) / 2.0, min=0.0, max=1.0), caption=f"blur_mask_decode_target_latent"),
                        'Image/blured_mask_pred_x0':        wandb.Image(torch.clamp((mask_blur_decode_pred_x0[0]       + 1.0) / 2.0, min=0.0, max=1.0), caption=f"blur_mask_decode_pred_x0"),
                    }, step=iter)
            # endregion
        wb_run.finish()
        # latent_loss_all.append(latent_loss_index)
        # latent_x0_loss_all.append(latent_x0_loss_index)
        # img_loss_all.append(img_loss_index)
        # img_loss_x0_all.append(img_loss_x0_index)
        # decode_loss_all.append(decode_loss_index)
    # region plotting
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(8,10))
    # # fig, ax = plt.subplots(figsize=(10,10))
    # x = np.array(azi_list) - int(np.rad2deg(gt_azimuth))
    # ax.plot(x[::1], latent_loss_all[0][::1], 'ro-', label='latent_loss')
    # ax.set_xlabel('Azimuth Error', fontsize=14)
    # ax.set_ylabel('Loss', fontsize=14)
    # ax.legend(
    #     loc='best',
    #     fontsize=14,
    #     shadow=False,
    #     facecolor='#bbb',
    #     edgecolor='#000',
    #     title=f'Loss Curve @ index{max_index}',
    #     title_fontsize=14)
    # # fig.savefig('../runs/imgs/lossCurve_latent_loss.png')
    # tb_writer.add_figure('LossCurve/latent_pred_target', fig, 1)
    # ax.cla()

    # ax.plot(x[::1], latent_x0_loss_all[0][::1], 'mo-', label='latent_loss@t0')
    # ax.set_xlabel('Azimuth Error', fontsize=14)
    # ax.set_ylabel('Loss', fontsize=14)
    # ax.legend(
    #     loc='best',
    #     fontsize=14,
    #     shadow=False,
    #     facecolor='#bbb',
    #     edgecolor='#000',
    #     title=f'Loss Curve @ index{max_index}',
    #     title_fontsize=14)
    # # fig.savefig('../runs/imgs/lossCurve_latent_t0_loss.png')
    # tb_writer.add_figure('LossCurve/latent_t0', fig, 1)
    # ax.cla()

    # ax.plot(x[::1], img_loss_all[0][::1], 'bo-', label='img_loss')
    # ax.set_xlabel('Azimuth Error', fontsize=14)
    # ax.set_ylabel('Loss', fontsize=14)
    # ax.legend(
    #     loc='best',
    #     fontsize=14,
    #     shadow=False,
    #     facecolor='#bbb',
    #     edgecolor='#000',
    #     title=f'Loss Curve @ index{max_index}',
    #     title_fontsize=14)
    # # fig.savefig('../runs/imgs/lossCurve_img_loss.png')
    # # tb_writer.add_figure('LossCurve/img_pred_target', fig, 1)
    # ax.cla()

    # ax.plot(x[::1], img_loss_x0_all[0][::1], 'co-', label='img_loss@t0')
    # ax.set_xlabel('Azimuth Error', fontsize=14)
    # ax.set_ylabel('Loss', fontsize=14)
    # ax.legend(
    #     loc='best',
    #     fontsize=14,
    #     shadow=False,
    #     facecolor='#bbb',
    #     edgecolor='#000',
    #     title=f'Loss Curve @ index{max_index}',
    #     title_fontsize=14)
    # fig.savefig('../runs/imgs/lossCurve_img_t0_loss.png')
    # tb_writer.add_figure('LossCurve/img_t0', fig, 1)
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
    gt_elev_deg = 0
    gt_azi_deg = 0
    match1 = re.search(r"elev=(-?[\d.]+)_azi=(-?[\d.]+)", ref_image_path)
    match2 = re.search(r"elev=(-?[\d.]+)_azi=(-?[\d.]+)", target_image_path)
    if match1 and match2:
        gt_elev_deg = float(match2.group(1)) - float(match1.group(1))
        gt_azi_deg = float(match2.group(2)) - float(match1.group(2))
        if gt_azi_deg > 180:
            gt_azi_deg -= 360
        print(f"start_rel: elev= {rel_elev_deg}, azi= {rel_azi_deg} | gt_rel: elev= {gt_elev_deg}, azi= {gt_azi_deg}")
    # endregion

    main_run(conf = conf,
             input_im = ref_image, target_im = target_image,
             models = models, device = device,
             learning_rate = conf.model.lr,
             gt_elevation = np.deg2rad(gt_elev_deg),
             gt_azimuth = np.deg2rad(gt_azi_deg),
             gt_radius = 0.0,
             preprocess=True,
             start_elevation = np.deg2rad(rel_elev_deg),
             start_azimuth = np.deg2rad(rel_azi_deg),
             start_radius = conf.input.rel_radius,
             n_samples= conf.model.n_samples,
             scale= conf.model.guide_scale,
             ddim_eta= conf.model.ddim_eta,
            )