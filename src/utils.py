import numpy as np
import torch

def cartesian_to_spherical(xyz):

    xy = xyz[0]**2 + xyz[1]**2
    radius = np.sqrt(xy + xyz[2]**2)
    theta = np.arctan2(np.sqrt(xy), xyz[2])
    azimuth = np.arctan2(xyz[1], xyz[0])

    return np.array([theta, azimuth, radius])

def relative_spherical(xyz_target, xyz_cond):

    sp_target = cartesian_to_spherical(xyz_target)
    sp_cond = cartesian_to_spherical(xyz_cond)

    theta_cond, azimuth_cond, z_cond = sp_cond
    theta_target, azimuth_target, z_target = sp_target

    d_theta = theta_target - theta_cond
    d_azimuth = (azimuth_target - azimuth_cond) % (2 * np.pi)
    d_z = z_target - z_cond

    return np.array([d_theta, d_azimuth, d_z])

def elu_to_c2w(eye, lookat, up):
    with np.errstate(divide='ignore',invalid='ignore'):
        if isinstance(eye, list):
            eye = np.array(eye)
        if isinstance(lookat, list):
            lookat = np.array(lookat)
        if isinstance(up, list):
            up = np.array(up)

        l = eye - lookat
        l = l / np.linalg.norm(l)
        s = np.cross(l, up)
        s = s / np.linalg.norm(s)
        uu = np.cross(s, l)

        rot = np.eye(3)
        rot[0, :] = -s
        rot[1, :] = uu
        rot[2, :] = l
        
        c2w = np.eye(4)
        c2w[:3, :3] = rot.T
        c2w[:3, 3] = eye

        return c2w

def spherical_to_cartesian(sph):

    theta, azimuth, radius = sph

    return np.array([
        radius * np.sin(theta) * np.cos(azimuth),
        radius * np.sin(theta) * np.sin(azimuth),
        radius * np.cos(theta),
    ])


def translate(x, y, z):
    return np.array(
        [
            [1,  0,  0,  x], 
            [0,  1,  0,  y], 
            [0,  0,  1,  z], 
            [0,  0,  0,  1]
        ],
        dtype=np.float32,
    )


def rotate_x(a):
    s, c = np.sin(a), np.cos(a)
    return np.array(
        [
            [1,  0,  0,  0], 
            [0,  c,  s,  0], 
            [0, -s,  c,  0], 
            [0,  0,  0,  1]
        ],
        dtype=np.float32,
    )


def rotate_y(a):
    s, c = np.sin(a), np.cos(a)
    return np.array(
        [
            [c,  0,  s,  0],
            [0,  1,  0,  0], 
            [-s, 0,  c,  0], 
            [0,  0,  0,  1]
        ],
        dtype=np.float32,
    )


def scale(s):
    return np.array(
        [
            [s,  0,  0,  0],
            [0,  s,  0,  0], 
            [0,  0,  s,  0], 
            [0,  0,  0,  1]
        ],
        dtype=np.float32,
    )


def sph2mat(sph):
    theta, azimuth, radius = sph
    mv = translate(0, 0, -radius) @ rotate_x(theta) @ rotate_y(-azimuth)
    w2c = mv
    c2w = np.linalg.inv(mv)
    return c2w, w2c

# def mat2sph(T, in_deg=False, return_radius=False):
#     if len(T.shape) == 2:
#         T = T.unsqueeze(0)
#     xyz = T[:, :3, 3]
#     radius = torch.norm(xyz, dim=1, keepdim=True)
#     xyz = xyz / radius
#     theta = -torch.asin(xyz[:, 1])
#     azimuth = torch.atan2(xyz[:, 0], xyz[:, 2])

#     if in_deg:
#         theta, azimuth = theta.rad2deg(), azimuth.rad2deg()
#     if return_radius:
#         return torch.stack((theta, azimuth, radius.squeeze(0))).T.numpy()
#     return torch.stack((theta, azimuth)).T.numpy()

def c2w_to_elu(c2w):

    w2c = np.linalg.inv(c2w)
    eye = c2w[:3, 3]
    lookat_dir = -w2c[2, :3]
    lookat = eye + lookat_dir
    up = w2c[1, :3]

    return eye, lookat, up

def compute_angular_error(pred_rel_sph, gt_rel_sph, radius=0.35):
    # Scaling relative radius from the zero123 scale to the actual scale.
    # The scale range of zero123 is (1.5, 2.2), we use the average value 1.85 as the zero123 scale.
    pred_rel_sph[2] += radius
    gt_rel_sph[2] += radius
    pred_rel_sph[2] = pred_rel_sph[2] * radius / 1.85

    pred_c2w, pred_w2c = sph2mat(pred_rel_sph)
    # pred_w2c = np.linalg.inv(pred_c2w)

    gt_c2w, gt_w2c = sph2mat(gt_rel_sph)
    # gt_w2c = np.linalg.inv(gt_c2w)

    pred_xyz = pred_c2w[:3, 3]
    gt_xyz = gt_c2w[:3, 3]
    dist = np.linalg.norm(pred_xyz - gt_xyz, 2) / radius

    target_rot = gt_w2c[:3, :3]
    pred_rot = pred_w2c[:3, :3]

    R_rel = pred_rot.T @ target_rot

    tr = (np.trace(R_rel) - 1) / 2
    theta = np.arccos(tr.clip(-1, 1))

    return dist, np.rad2deg(theta)