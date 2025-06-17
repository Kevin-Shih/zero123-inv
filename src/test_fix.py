import os
import argparse
import yaml
import time
import re
import wandb
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from rich import print
from einops import rearrange
from lovely_numpy import lo
from contextlib import nullcontext
from tqdm import tqdm

# from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import create_carvekit_interface, load_and_preprocess, instantiate_from_config
from torch import Tensor, optim, nn
from torch.nn.parameter import Parameter
from torch.amp.autocast_mode import autocast
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

    input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
    input_im = input_im * 2 - 1 # move to [-1, 1]
    input_im = transforms.Resize([h, w])(input_im)
    return input_im, forground_mask

def calculate_param_ddim(sampler, index, n_samples, device):
    alphas = sampler.ddim_alphas
    alphas_prev = sampler.ddim_alphas_prev
    sigmas = sampler.ddim_sigmas
    sqrt_one_minus_alphas = sampler.ddim_sqrt_one_minus_alphas
    # select parameters corresponding to the currently considered timestep
    a_t = torch.full((n_samples, 1, 1, 1), alphas[index], device=device)
    a_prev = torch.full((n_samples, 1, 1, 1), alphas_prev[index], device=device)
    sigma_t = torch.full((n_samples, 1, 1, 1), sigmas[index], device=device)
    sqrt_one_minus_at = torch.full((n_samples, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)
    return a_t, a_prev, sigma_t, sqrt_one_minus_at

def sample_model(input_im, target_im, LDModel, precision, h, w,
                 elevation, azimuth, radius, n_samples,
                 scale = 3.0, ddim_steps= 75, ddim_eta= 0.15, index = 5):
    step = int(1000//ddim_steps) * max(index, 0) + 1
    step_target_inter = int(1000//ddim_steps) if index > 0 else 1
    precision_scope = autocast if precision == 'autocast' else nullcontext
    sampler = DDIMSampler(LDModel)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_discretize="uniform", ddim_eta=ddim_eta, verbose=False)
    with precision_scope('cuda'):
        # with LDModel.ema_scope():
            # region input/condition
            # Set time step and noisy latent shape
            t = torch.full((n_samples,), step, device=input_im.device, dtype=torch.long)
            size = (n_samples, 4, h // 8, w // 8)
            # Get input & target latent
            input_encoder_posterior = LDModel.encode_first_stage(input_im)
            input_im_z = LDModel.get_first_stage_encoding(input_encoder_posterior)
            target_encoder_posterior = LDModel.encode_first_stage(target_im)
            target_im_z = LDModel.get_first_stage_encoding(target_encoder_posterior)
            # Add noise to the input latent and target latent
            _noise = torch.randn_like(input_im_z)
            input_latent = LDModel.q_sample(target_im_z.clone().detach(), t, _noise) # perfecInput
            # input_latent = LDModel.q_sample(input_im_z, t, _noise) # fix input
            # _target_start_latent = LDModel.q_sample(target_im_z.clone().detach(), t, _noise)
            target_latent = LDModel.q_sample(target_im_z, t-step_target_inter, _noise)

            # latent_diff = _target_start_latent - target_latent
            # latent_x0_diff = _target_start_latent - target_im_z
            # Get condintioning
            img_cond = LDModel.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = torch.cat([elevation, torch.sin(azimuth), torch.cos(azimuth), radius])
            T_batch = T[None, None, :].repeat(n_samples, 1, 1)
            c = torch.cat([img_cond, T_batch], dim=-1).float()
            c_proj = LDModel.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c_proj]
            cond['c_concat'] = [input_encoder_posterior.mode().detach().repeat(n_samples, 1, 1, 1)]
            uc = {}
            uc['c_concat'] = [torch.zeros(size, device=img_cond.device)]
            uc['c_crossattn'] = [torch.zeros_like(c_proj, device=img_cond.device)]
            # endregion

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

            # a_t, a_prev, sigma_t, sqrt_one_minus_at = calculate_param_ddim(sampler, index, n_samples, img_cond.device)
            a_t, a_prev, sigma_t, sqrt_one_minus_at = sampler.calculate_param_ddim(index, n_samples, img_cond.device)
            pred_x0 = (input_latent - sqrt_one_minus_at * e_t) / a_t.sqrt() # current prediction for x_0
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t # direction pointing to x_t
            pred_x_idx_minus_one = a_prev.sqrt() * pred_x0 + dir_xt + sigma_t * _noise

            return pred_x_idx_minus_one, pred_x0, input_latent, target_latent, target_im_z

class PoseT(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pose):

        p1 = pose[..., 0:1]
        p2 = torch.sin(pose[..., 1:2])
        p3 = torch.cos(pose[..., 1:2])
        p4 = pose[..., 2:3]

        return torch.cat([p1, p2, p3, p4], dim=-1)

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
    loss, _ = model.shared_step(batch, ts=np.arange(mn, mx, (mx-mn) / bsz), noise=noise[:bsz])

    return loss


def idp_pairwise_loss(pose, model, cond_image, target_image, ts_range, probe_bsz, noise=None):

    theta, azimuth, radius = pose

    pose1 = torch.tensor([[theta, azimuth, radius]], device=model.device, dtype=torch.float32)
    pose2 = torch.tensor([[-theta, np.pi*2-azimuth, -radius]], device=model.device, dtype=torch.float32)
    loss1 = idp_noise_loss(model, cond_image, target_image, pose1, ts_range, probe_bsz, noise=noise)
    loss2 = idp_noise_loss(model, target_image, cond_image, pose2, ts_range, probe_bsz, noise=noise)

    return loss1 + loss2


def main_run(conf,
             input_im, target_im,
             models, device,
             gt_elevation=0.0, gt_azimuth=0.0, gt_radius=0.0,
             start_elevation=0.0, start_azimuth=0.0, start_radius=0.0,
             preprocess=True,
             scale=3.0, n_samples=1, ddim_steps=75, ddim_eta=0.15,
             learning_rate = 1e-3,
             precision='fp32', h=256, w=256,):
    '''
    :param raw_im (PIL Image).
    '''

    input_im.thumbnail([256, 256], Image.Resampling.LANCZOS)
    target_im.thumbnail([256, 256], Image.Resampling.LANCZOS)

    input_im, _ = preprocess_image(models, input_im, preprocess, h=h, w=w, device=device)
    target_im, target_mask = preprocess_image(models, target_im, preprocess, h=h, w=w, device=device)
    target_mask = Tensor(target_mask).to(device)
    wb_run.log({'Image/input_im':  wandb.Image(torch.clamp((input_im + 1.0) / 2.0, min=0.0, max=1.0)[0],  caption=f"input_im"),
                'Image/target_im': wandb.Image(torch.clamp((target_im + 1.0) / 2.0, min=0.0, max=1.0)[0], caption=f"target_im"),
    }, step=0)
    """ NSFW content detection
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

    LDModel = models['turncam']

    # used_x = -x  # NOTE: Polar makes more sense in Basile's opinion this way!
    # used_elevation = elevation  # NOTE: Set this way for consistency.
    est_elev = Parameter(data=Tensor([start_elevation]).to(torch.float32).to(device), requires_grad=True)
    est_azimuth = Parameter(data=Tensor([start_azimuth]).to(torch.float32).to(device), requires_grad=True)
    est_radius = Parameter(data=Tensor([start_radius]).to(torch.float32).to(device), requires_grad=True)

    print("learning_rate = ", learning_rate)
    # optimizer = optim.Adam([{'params': est_elev},
    #                         {'params': est_azimuth}], lr=learning_rate)#{'params': est_radius, 'lr': 1e-18}
    optimizer = optim.Adam([{'params': est_azimuth, 'param_names': 'azi'}], lr=learning_rate)

    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=conf.model.lr_scheduler.milestones,
    #                                            gamma=conf.model.lr_scheduler.gamma)

    max_iter = conf.model.iters
    pbar = tqdm(range(max_iter), desc='DDIM', total=max_iter, ncols=140)
    max_index = conf.input.max_index
    min_index = max_index if conf.input.min_index is None else max(conf.input.min_index, 0)
    idx_decrease_interval = max_iter / (max_index - min_index + 1)
    for i, iter in enumerate(pbar, start=1):
        pbar.set_description_str(f'[{iter}/{max_iter}]')
        optimizer.zero_grad()

        """  random index
        if iter < 200:
            index = np.random.randint(20, max_index)
        elif iter < 500:
            index = np.random.randint(10, 20)
        else:
            index = np.random.randint(min_index, 10)
        """
        index = int(max_index - iter//idx_decrease_interval)
        pred_target, pred_x0, input_latent, target_latent, target_latent_x0 = sample_model(input_im, target_im, LDModel, precision,
                                                    h, w, est_elev, est_azimuth, est_radius, n_samples= n_samples, scale= scale,
                                                    ddim_steps= ddim_steps, ddim_eta= ddim_eta, index= index)
        
        # idp_pose=torch.cat([est_elev, est_azimuth, est_radius], dim =-1).to(device)
        # idp_single_loss = idp_noise_loss(LDModel, raw_input_im, raw_target_im, idp_pose,
        #                                 ts_range=[0.2, 0.21], bsz=1, noise=None)
        # idp_pair_loss = idp_pairwise_loss(idp_pose, LDModel, rearrange(target_im,'b c h w -> b h w c'), rearrange(input_im,'b c h w -> b h w c'), ts_range=[0.2, 0.21], probe_bsz=1, noise=None)
        batch = {}
        bsz = 1
        raw_input_im = rearrange(input_im,'b c h w -> b h w c')
        raw_target_im = rearrange(target_im,'b c h w -> b h w c')
        batch['image_target'] = raw_target_im.repeat(bsz, 1, 1, 1)
        batch['image_cond'] = raw_input_im.repeat(bsz, 1, 1, 1)
        batch['T'] = torch.cat([est_elev, torch.sin(est_azimuth), torch.cos(est_azimuth), est_radius]).repeat(bsz, 1)
        idp_single_loss, _ = LDModel.shared_step(batch, ts=np.arange(0.2, 0.21, (0.01) / bsz), noise=torch.randn(bsz, 4, 32, 32, device=LDModel.device)[:bsz])
        
        decode_pred_target   = LDModel.decode_first_stage(pred_target)
        decode_pred_x0       = LDModel.decode_first_stage(pred_x0)
        # decode_target_latent = LDModel.decode_first_stage(target_latent)

        if conf.model.clamp is not None:
            print(f'clamping output: {conf.model.clamp}')
            if conf.model.clamp == 'normal':
                decode_pred_target   = torch.clamp((decode_pred_target   + 1.0) / 2.0, min=0.0, max=1.0)
                # decode_target_latent = torch.clamp((decode_target_latent + 1.0) / 2.0, min=0.0, max=1.0)
                decode_pred_x0       = torch.clamp((decode_pred_x0       + 1.0) / 2.0, min=0.0, max=1.0)
            elif conf.model.clamp == 'ddim':
                decode_pred_target   = torch.clamp(decode_pred_target  , min=-1.0, max=1.0)
                # decode_target_latent = torch.clamp(decode_target_latent, min=-1.0, max=1.0)
                decode_pred_x0       = torch.clamp(decode_pred_x0      , min=-1.0, max=1.0)

        # pred_latent_diff    = input_latent - pred_target
        # pred_latent_x0_diff = input_latent - pred_x0
        # noise_loss          = nn.functional.mse_loss(pred_latent_diff, latent_diff) # loss between latent_diff and denoised diff
        # noise_x0_loss       = nn.functional.mse_loss(pred_latent_x0_diff, latent_x0_diff) # loss between latent_x0_diff and denoised x0 diff

        blur = transforms.GaussianBlur(kernel_size=[conf.model.blur_k_ize], sigma = (conf.model.blur_min, conf.model.blur_max))
        blur_decode_pred_target     = blur(decode_pred_target)
        blur_decode_pred_x0         = blur(decode_pred_x0)
        # blur_decode_target_latent   = blur(decode_target_latent)
        blur_target_im              = blur(target_im)
        
        mask_blur_decode_pred_target     = blur_decode_pred_target   * target_mask.float()
        mask_blur_decode_pred_x0         = blur_decode_pred_x0       * target_mask.float()
        # mask_blur_decode_target_latent   = blur_decode_target_latent * target_mask.float()
        mask_blur_target_im              = blur_target_im            * target_mask.float()

        latent_loss    = nn.functional.mse_loss(pred_target, target_latent)
        latent_x0_loss = nn.functional.mse_loss(pred_x0, target_latent_x0.expand_as(pred_x0))
        # img_loss       = nn.functional.mse_loss(decode_pred_target, decode_target_latent)
        # img_x0_loss    = nn.functional.mse_loss(decode_pred_x0, target_im.expand_as(blur_decode_pred_x0))
        # blur_img_loss       = nn.functional.mse_loss(blur_decode_pred_target, blur_decode_target_latent)
        blur_img_x0_loss    = nn.functional.mse_loss(blur_decode_pred_x0, blur_target_im.expand_as(blur_decode_pred_x0))

        no_reduct_mse = nn.MSELoss(reduction='none')
        non_zero_elements = target_mask.sum()
        # _mask_img_loss = (no_reduct_mse(decode_pred_target, decode_target_latent) * target_mask.float()).sum()
        # mask_img_loss = _mask_img_loss / non_zero_elements
        # _mask_img_x0_loss = (no_reduct_mse(decode_pred_x0, target_im.expand_as(decode_pred_x0)) * target_mask.float()).sum()
        # mask_img_x0_loss = _mask_img_x0_loss / non_zero_elements
        
        # _blur_mask_img_loss = (no_reduct_mse(blur_decode_pred_target, blur_decode_target_latent) * target_mask.float()).sum()
        # blur_mask_img_loss = _blur_mask_img_loss / non_zero_elements
        _blur_mask_img_x0_loss = (no_reduct_mse(blur_decode_pred_x0, blur_target_im.expand_as(blur_decode_pred_x0)) * target_mask.float()).sum()
        blur_mask_img_x0_loss = _blur_mask_img_x0_loss / non_zero_elements

        loss = idp_single_loss
        if conf.model.loss_amp:
            loss = loss * conf.model.loss_amp
        loss.backward()
        # toPil = transforms.ToPILImage()
        if conf.log_all_img and i % conf.log_all_img_freq == 0:
            # region detail logging
            with torch.no_grad():
                decode_input_latent  = LDModel.decode_first_stage(input_latent)
                decode_target_x0     = LDModel.decode_first_stage(target_latent_x0)

                if conf.model.clamp is not None:
                    print(f'clamping output: {conf.model.clamp}')
                    if conf.model.clamp == 'normal':
                        decode_input_latent  = torch.clamp((decode_input_latent  + 1.0) / 2.0, min=0.0, max=1.0)
                        decode_target_x0     = torch.clamp((decode_target_x0     + 1.0) / 2.0, min=0.0, max=1.0)
                    elif conf.model.clamp == 'ddim':
                        decode_input_latent  = torch.clamp(decode_input_latent , min=-1.0, max=1.0)
                        decode_target_x0     = torch.clamp(decode_target_x0    , min=-1.0, max=1.0)

                input_latent_loss = nn.functional.mse_loss(input_latent, target_latent)
                decode_loss       = nn.functional.mse_loss(decode_target_x0, target_im)

                wb_run.log({
                    'Loss/input_latent':    input_latent_loss.item(),
                    'Loss/decode':          decode_loss.item(),
                    'Image/input_latent':   wandb.Image(decode_input_latent[0]  , caption=f"generated_input_latent"),
                    'Image/target_x0':      wandb.Image(decode_target_x0[0]     , caption=f"generated_target_x0"),
                }, step=iter)
            # endregion

        optimizer.step()
        # if conf.model.lr_scheduler.use:
        #     scheduler.step()
        with torch.no_grad():
            temp_elev= np.rad2deg(est_elev.item())
            temp_azi= np.rad2deg(est_azimuth.item())

            err = [temp_elev - np.rad2deg(gt_elevation), temp_azi - np.rad2deg(gt_azimuth)]
            pbar.set_postfix_str(f'step: {index}-{int(1000//ddim_steps) * max(index, 0) + 1}, ' +
                                 f'loss: {loss.item():.3f}, Err elev, azi= {err[0]:.2f}, {err[1]:.2f}, ' +
                                 f'Curr= {temp_elev:.2f}, {temp_azi:.2f}')
            wb_run.log({
                "Estimate/elevation":           temp_elev,
                "Estimate/azimuth":             temp_azi,
                'Error/elevation':              err[0],
                'Error/azimuth':                err[1],
                'Error/Abs elevation':          np.abs(err[0]),
                'Error/Abs azimuth':            np.abs(err[1]),
                'Log/ddim_index':               index,
                'Loss/total':                   loss.item(),
                # 'Loss/img':                     img_loss.item(),
                # 'Loss/img_x0':                  img_x0_loss.item(),
                # 'Loss/mask_img':                mask_img_loss.item(),
                # 'Loss/mask_img_x0':             mask_img_x0_loss.item(),
                # 'Loss/blur_img':                blur_img_loss.item(),
                'Loss/blur_img_x0':             blur_img_x0_loss.item(),
                # 'Loss/blur_mask_img':           blur_mask_img_loss.item(),
                'Loss/blur_mask_img_x0':        blur_mask_img_x0_loss.item(),
                # 'Loss/noise_loss':              noise_loss.item(),
                # 'Loss/noise_x0_loss':           noise_x0_loss.item(),
                'Loss/latent':                  latent_loss.item(),
                'Loss/latent_x0':               latent_x0_loss.item(),
                # 'Loss/neg_latent_x0':           -1*latent_x0_loss.item(),
                'Loss/idp_single_loss':         idp_single_loss,
                # 'Loss/idp_pair_loss':           idp_pair_loss,
                # 'Image/blured_target_im':           wandb.Image(blur_target_im[0]                   , caption=f"blured_target_im"),
                'Image/blured_mask_target_im':      wandb.Image(mask_blur_target_im[0]              , caption=f"blur_mask_target_im"),
                'Image/pred_target':                wandb.Image(decode_pred_target[0]               , caption=f"generated_pred_target"),
                # 'Image/target_latent':              wandb.Image(decode_target_latent[0]             , caption=f"generated_target_latent"),
                'Image/pred_x0':                    wandb.Image(decode_pred_x0[0]                   , caption=f"generated_pred_x0"),
                'Image/blured_pred_target':         wandb.Image(blur_decode_pred_target[0]          , caption=f"blured_pred_target"),
                # 'Image/blured_target_latent':       wandb.Image(blur_decode_target_latent[0]        , caption=f"blured_target_latent"),
                'Image/blured_pred_x0':             wandb.Image(blur_decode_pred_x0[0]              , caption=f"blured_pred_x0"),
                'Image/blured_mask_pred_target':    wandb.Image(mask_blur_decode_pred_target[0]     , caption=f"blur_mask_decode_pred_target"),
                # 'Image/blured_mask_target_latent':  wandb.Image(mask_blur_decode_target_latent[0]   , caption=f"blur_mask_decode_target_latent"),
                'Image/blured_mask_pred_x0':        wandb.Image(mask_blur_decode_pred_x0[0]         , caption=f"blur_mask_decode_pred_x0"),
            }, step=iter)
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
    conf = OmegaConf.load(args.config)
    print(f'Loading configs from {os.path.basename(args.config)}...')

    ref_image_path = os.path.join(conf.dataroot, conf.input.ref_image)
    target_image_path = os.path.join(conf.dataroot, conf.input.target_image)
    rel_elev_deg = conf.input.rel_elev
    rel_azi_deg = conf.input.rel_azi
    rel_radius = conf.input.rel_radius
    device = f"cuda:{conf.model.gpu_idx}"
    model_config_obj = OmegaConf.load(conf.model.model_config)

    assert torch.cuda.is_available()
    assert os.path.exists(conf.model.ckpt)
    assert os.path.exists(ref_image_path)
    # assert conf.model.lr_scheduler

    # region Load image and check gt.
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

    # region Instantiate all models beforehand for efficiency.
    models = dict()
    print('Instantiating LatentDiffusion...', end='\r')
    models['turncam'] = load_model_from_config(model_config_obj, conf.model.ckpt, device=device, verbose=True)
    print('Instantiating Carvekit HiInterface...', end='\r')
    models['carvekit'] = create_carvekit_interface()
    """
    print('Instantiating StableDiffusionSafetyChecker...', end='\r')
    models['nsfw'] = StableDiffusionSafetyChecker.from_pretrained(
        'CompVis/stable-diffusion-safety-checker').to(device)
    """
    print('Instantiating AutoFeatureExtractor...', end='\r')
    models['clip_fe'] = AutoFeatureExtractor.from_pretrained(
        'CompVis/stable-diffusion-safety-checker')
    # endregion

    # region summary setup
    curr_time = time.localtime(time.time())
    mon = curr_time.tm_mon
    mday = curr_time.tm_mday
    hours = curr_time.tm_hour
    mins = curr_time.tm_min + curr_time.tm_sec / 60
    wb_run = wandb.init(
        entity="kevin-shih",
        project="Zero123-Adv",
        group= f'{conf.group_name}',
        name= f'{conf.run_name}_gt-{gt_elev_deg:.0f}-{gt_azi_deg:.0f}',
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

    main_run(conf = conf,
             input_im = ref_image, target_im = target_image,
             models = models, device = device,
             learning_rate = conf.model.lr,
             gt_elevation = np.deg2rad(gt_elev_deg),
             gt_azimuth = np.deg2rad(gt_azi_deg),
             gt_radius = 0.0,
             start_elevation = np.deg2rad(rel_elev_deg),
             start_azimuth = np.deg2rad(rel_azi_deg),
             start_radius = conf.input.rel_radius,
             n_samples = conf.model.n_samples,
             scale = conf.model.guide_scale,
             ddim_eta = conf.model.ddim_eta,
             preprocess = True
            )
    wb_run.finish()