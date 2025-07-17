import os
import argparse
import time
import wandb
import itertools
import torch
import numpy as np
from omegaconf import OmegaConf
from rich import print
from contextlib import nullcontext
from tqdm import tqdm
import multiprocessing

from ldm.models.diffusion.ddim import DDIMSampler
from torch import Tensor, optim, nn
from torch.nn.parameter import Parameter
from torch.amp.autocast_mode import autocast
from torchvision import transforms
from utils.pose import compute_pose_error, sph2mat, mat2sph
from utils.util import mask_resize, set_random_seed, load_models, load_image, load_img_and_gt, split_list

def eval_pose(transform_fp, gt_transform_fp, image_dir, id, **kwargs):
    camtoworlds = load_img_and_gt(image_dir, transform_fp, verbose=False)[1]
    print(f'est_c2w   1 = {mat2sph(camtoworlds[1], in_deg=True, return_radius=True)[0]}')
    gt_camtoworlds = load_img_and_gt(image_dir, gt_transform_fp, verbose=False)[1]
    gt_camtoworlds = gt_camtoworlds[id]
    print(f'gt_c2w    1 = {mat2sph(gt_camtoworlds[1], in_deg=True, return_radius=True)[0]}')

    pose_err = [compute_pose_error(pred, gt) for pred, gt in zip(camtoworlds[1:], gt_camtoworlds[1:])]
    pose_err = np.array(pose_err).mean(axis=0)
    print(f"Rot. error: {pose_err[0]:.2f}, Trans. error: {pose_err[1]:.2f}")
    return pose_err

def sample_model(ref_im, target_im, LDModel, sampler, elevation, azimuth,
                 radius, n_samples, ddim_steps= 75, index = 5, precision = 'fp32'):
    step = int(1000//ddim_steps) * max(index, 0) + 1
    step_target_inter = int(1000//ddim_steps) if index > 0 else 1
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with LDModel.ema_scope():
            # region input/condition
            # Set time step and noisy latent shape
            t = torch.full((n_samples,), step, device=ref_im.device, dtype=torch.long)
            # Get input & target latent
            input_encoder_posterior = LDModel.encode_first_stage(ref_im)
            ref_im_z = LDModel.get_first_stage_encoding(input_encoder_posterior)
            target_encoder_posterior = LDModel.encode_first_stage(target_im)
            target_im_z = LDModel.get_first_stage_encoding(target_encoder_posterior)
            # Add noise to the input latent and target latent
            _noise = torch.randn_like(ref_im_z)
            input_latent = LDModel.q_sample(target_im_z.clone().detach(), t, _noise) # perfecInput
            target_latent = LDModel.q_sample(target_im_z, t-step_target_inter, _noise)
            # Get condintioning
            img_cond = LDModel.get_learned_conditioning(ref_im).tile(n_samples, 1, 1)
            radius = torch.sin(radius-0.35) * 0.8 # search2
            T = torch.cat([elevation, torch.sin(azimuth), torch.cos(azimuth), radius])
            T_batch = T[None, None, :].repeat(n_samples, 1, 1)
            c = torch.cat([img_cond, T_batch], dim=-1)
            c_proj = LDModel.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c_proj]
            cond['c_concat'] = [input_encoder_posterior.mode().detach().repeat(n_samples, 1, 1, 1)]
            # endregion
            e_t = LDModel.apply_model(input_latent, t, cond)

            # a_t, a_prev, sigma_t, sqrt_one_minus_at = calculate_param_ddim(sampler, index, n_samples, img_cond.device)
            a_t, a_prev, sigma_t, sqrt_one_minus_at = sampler.calculate_param_ddim(index, n_samples, img_cond.device)
            pred_x0 = (input_latent - sqrt_one_minus_at * e_t) / a_t.sqrt() # current prediction for x_0
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t # direction pointing to x_t
            pred_x_idx_minus_one = a_prev.sqrt() * pred_x0 + dir_xt + sigma_t * _noise

            noise_loss = torch.nn.functional.mse_loss(LDModel.apply_model(input_latent, t, cond), _noise, reduction='none').mean([1, 2, 3])
            logvar_t = LDModel.logvar[t].to(LDModel.device)
            noise_loss = noise_loss / torch.exp(logvar_t) + logvar_t

            return pred_x_idx_minus_one, pred_x0, input_latent, target_latent, target_im_z, noise_loss.mean()

def pred_pose(models, conf):
    device = torch.device(f'cuda:{conf.gpu_idx}')
    ddim_steps = conf.args.ddim_steps
    mask_init_size = conf.args.mask_init_size
    lr_scheduler = conf.optim.lr_scheduler
    LDModel = models['turncam']
    #img, mask, c2w, pose
    ref_im, _, gt_ref_c2w, gt_ref_pose = load_img_and_gt(models, conf.data.image_dir, conf.data.gt_transform_fp, idx=conf.data.idx[0], device=device)
    # ref_im, _ = load_image(models, conf.data.ref_image_path, device=device)
    target_im, _target_mask, gt_target_c2w, gt_target_pose = load_img_and_gt(models, conf.data.image_dir, conf.data.gt_transform_fp, idx=conf.data.idx[1], device=device)
    # target_im, _target_mask = load_image(models, conf.data.target_image_path, device=device)
    gt_rel_pose = gt_target_pose - gt_ref_pose
    print('gt_rel_pose', gt_rel_pose)  # in degrees
    gt_rel_pose[:2] = np.deg2rad(gt_rel_pose[:2])
    target_mask = mask_resize(_target_mask, size=int(256 * mask_init_size))
    mask_size_step = (mask_init_size - 1) / int(conf.optim.iters / 20)
    
    est_elev = Parameter(data=Tensor([0]).to(torch.float32).to(device), requires_grad=True)
    est_azimuth = Parameter(data=Tensor([0]).to(torch.float32).to(device), requires_grad=True)
    est_radius = Parameter(data=Tensor([0]).to(torch.float32).to(device), requires_grad=True)

    optimizer = optim.Adam([{'params': est_elev, 'param_names': 'elev'},
                            {'params': est_azimuth, 'param_names': 'azi'},
                            {'params': est_radius}], lr=conf.optim.lr)
    if lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor= lr_scheduler.gamma, patience= lr_scheduler.patience)

    max_iter = conf.optim.iters
    pbar = tqdm(range(max_iter), desc='DDIM', total=max_iter, ncols=140)
    max_index = conf.input.max_index
    min_index = max_index if conf.input.min_index is None else max(conf.input.min_index, 0)
    idx_decrease_interval = max_iter / (max_index - min_index + 1)
    dist_err, angular_err, temp_dist = 0, 0, 0
    sampler = DDIMSampler(LDModel)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_discretize="uniform", ddim_eta=conf.args.ddim_eta, verbose=False)
    for i, iter in enumerate(pbar, start=1):
        pbar.set_description_str(f'[{i}/{max_iter}]')
        optimizer.zero_grad()

        index = int(max_index - iter//idx_decrease_interval)
        pred_target, pred_x0, input_latent, target_latent,\
        target_latent_x0, noise_loss = sample_model(ref_im, target_im, LDModel, sampler, 
                                                    est_elev, est_azimuth, est_radius, 
                                                    n_samples= conf.args.n_samples, index= index,)
        decode_pred_x0       = LDModel.decode_first_stage(pred_x0)

        blur = transforms.GaussianBlur(kernel_size=[conf.args.blur_k_size], sigma = (conf.args.blur_min if conf.args.blur_min else conf.args.blur_max, conf.args.blur_max))
        blur_decode_pred_x0         = blur(decode_pred_x0)
        blur_target_im              = blur(target_im)

        no_reduct_mse = nn.MSELoss(reduction='none')
        non_zero_elements = target_mask.sum()

        _blur_mask_img_x0_loss = (no_reduct_mse(blur_decode_pred_x0, blur_target_im.expand_as(blur_decode_pred_x0)) * target_mask.float()).sum()
        blur_mask_img_x0_loss = _blur_mask_img_x0_loss / non_zero_elements

        loss = blur_mask_img_x0_loss
        loss.backward()

        optimizer.step()
        if lr_scheduler:
            scheduler.step(loss)
        with torch.no_grad():
            if iter % 20 == 0:
                mask_factor = int(iter / 20 + 1)
                target_mask = mask_resize(_target_mask, size=int(256*(mask_init_size - mask_factor * mask_size_step)))
            dist_err, angular_err, temp_dist = compute_pose_error(pred_rel_sph= [est_elev.item(), est_azimuth.item(), est_radius.item()], 
                                                gt_rel_sph= gt_rel_pose, radius= gt_ref_pose[2])
            temp_elev= np.rad2deg(est_elev.item())
            temp_azi= np.rad2deg(est_azimuth.item())

            err = [temp_elev - np.rad2deg(gt_rel_pose[0]), temp_azi - np.rad2deg(gt_rel_pose[1]), temp_dist - gt_rel_pose[2]]
            pbar.set_postfix_str(
                                 f'step: {index}-{int(1000//ddim_steps) * max(index, 0) + 1}, ' +
                                 f'lr: {optimizer.state_dict()["param_groups"][0]["lr"]:.3f}, ' +
                                 f'loss: {loss.item():.3f}, Err= {angular_err:.2f}, {err[2]:.2f} ' +
                                 f'Curr= {temp_elev:.2f}, {temp_azi:.2f}, {temp_dist:.2f}' )
    # dist_err, angular_err, temp_dist = compute_pose_error(pred_rel_sph= [est_elev.item(), est_azimuth.item(), est_radius.item()], 
    #                                                         gt_rel_sph= gt_rel_pose, radius= .35)
    estimate_pose = [est_elev.item(), est_azimuth.item(), est_radius.item()]
    estimate_T, _ = sph2mat(estimate_pose)
    estimate_pose[:2] = np.rad2deg(estimate_pose[:2])
    print(f"[INFO] Estimated pose: [{estimate_pose[0]:.1f},{estimate_pose[1]:.1f},{estimate_pose[2]:.1f}], Rot. error: {angular_err:.2f}, Dist. error:{dist_err:.2f}")
    return estimate_pose, estimate_T, [angular_err, dist_err]

def main(conf, objs, idxs, wb_run):

    models = load_models(OmegaConf.load(conf.model.model_config), conf.model.ckpt)
    metric = []
    for obj in objs:
        for idx in idxs:
            print(f"[INFO] Optimizing pose {obj}:{idx}")
            conf.data.obj = obj
            conf.data.idx = idx
            pose, T, pose_err = pred_pose(models, conf)
            metric.append(pose_err)
    metric = np.array(metric)
    np.savez(f"{conf.run_name}/pose_{conf.data.dataset_name}.npz", metric)
    rot_p25, rot_p50, rot_p75 = np.percentile(metric[:, 0], [25, 50, 75])
    trans_p25, trans_p50, trans_p75 = np.percentile(metric[:, 1], [25, 50, 75])
    wb_run.log({'Error/Rot. error (p25)':  rot_p25,
                'Error/Trans. error (p25)': trans_p25,
                'Error/Rot. error (median)':  rot_p50,
                'Error/Trans. error (median)': trans_p50,
                'Error/Rot. error (p75)':  rot_p75,
                'Error/Trans. error (p75)': trans_p75,
                'Recall/ <= 5': sum(metric[:, 0] <= 5) / len(metric),
                'Recall/ <= 15': sum(metric[:, 0] <= 15) / len(metric),
                'Recall/ <= 30': sum(metric[:, 0] <= 30) / len(metric),
    }, step=0)
    return 0
    

if __name__ == '__main__':
    # region loading conf
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/sanity_check.yaml",
        help="path to configs to load OmegaConf from"
    )
    args, extras = parser.parse_known_args()
    cli_conf = OmegaConf.from_cli(extras)
    yaml_conf = OmegaConf.load(args.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    print(f'Loading configs from {os.path.basename(args.config)}...')
    assert torch.cuda.is_available()
    assert os.path.exists(conf.model.ckpt)
    assert os.path.exists(conf.model.model_config)
    # endregion
    
    # region summary setup
    curr_time = time.localtime(time.time())
    mon, mday, hours = curr_time.tm_mon, curr_time.tm_mday, curr_time.tm_hour
    mins = curr_time.tm_min + curr_time.tm_sec / 60
    wb_run = wandb.init(
        dir="../wandb/eval",
        entity="kevin-shih",
        project="Zero123-Adv",
        group= f'{conf.group_name}',
        name= f'{conf.run_name}_{mday:02d}_{hours:02d}-{mins:4.1f}',
        settings=wandb.Settings(x_disable_stats=True),
        config={
                "start_date": f'{mon:02d}-{mday:02d}',
                "start_time": f'{hours:02d}-{mins:4.1f}',
                **OmegaConf.to_container(conf, resolve=True),
        },
    )
    # endregion
    
    set_random_seed(conf.seed)
    comb = list(itertools.combinations(range(3), 2))
    idxs = [list(c) for c in comb]
    objs = sorted(os.listdir(f"{conf.data.root}/{conf.data.dataset_name}"))[0:5]
    main(conf, objs, idxs, wb_run)
    
    wb_run.finish()