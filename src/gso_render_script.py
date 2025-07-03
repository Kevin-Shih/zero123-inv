"""Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.

Example usage:
    blender -b -P blender_script.py -- \
        --object_path my_object.glb \
        --output_dir ./views \
        --engine CYCLES \
        --scale 0.8 \
        --num_images 12 \
        --camera_dist 1.2

Here, input_model_paths.json is a json file containing a list of paths to .glb.
"""

import argparse
import json
from logging import root
import math
import os
from pickle import FALSE
import random
import sys
import time
import urllib.request
import uuid
from typing import Tuple
from mathutils import Vector, Matrix
import numpy as np
# from utils import spherical_to_cartesian, sph2mat

import bpy
from mathutils import Vector

def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

# function from https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    # R_bcam2cv = Matrix(
    #     ((1, 0,  0),
    #     (0, 1, 0),
    #     (0, 0, 1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # # Build the coordinate transform matrix from world to computer vision camera
    # R_world2cv = R_bcam2cv@R_world2bcam
    # T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2bcam[0][:] + (T_world2bcam[0],),
        R_world2bcam[1][:] + (T_world2bcam[1],),
        R_world2bcam[2][:] + (T_world2bcam[2],)
        ))
    return RT

def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    # uid = uuid.uuid4()
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    # get the absolute path
    local_path = os.path.abspath(local_path)
    return local_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Path to the gso data root",
    )
    parser.add_argument(
        "--object_name",
        type=str,
        default="Circo_Fish_Toothbrush_Holder",
        help="Name of the object file",
    )
    parser.add_argument(
        "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"]
    )
        
    args = parser.parse_args(sys.argv[1:])
    target_dir = os.path.join(args.data_root, "gso_myrendering", args.object_name)
    obj_path = os.path.join(args.data_root, "gso_object", args.object_name, "model.obj")
    os.makedirs(target_dir, exist_ok=True)
    assert os.path.exists(target_dir)
    assert os.path.exists(obj_path)
    reset_scene()

    print('===================', args.engine, '===================')
    scene = bpy.context.scene
    render = bpy.context.scene.render
    
    cam = scene.objects["Camera"]
    cam.location = (0, 0, -0.35)
    cam.rotation_euler = (0, np.deg2rad(180), np.deg2rad(180))
    cam.data.lens = 50
    cam.data.sensor_width = 36

    # setup lighting
    bpy.ops.object.light_add(type="AREA", location=(0, -0.5, -1), rotation=(np.deg2rad(155), 0, 0))
    bpy.data.lights["Area"].energy = 12

    render.engine = "CYCLES"
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.film_transparent = True
    render.resolution_x = 256
    render.resolution_y = 256
    render.resolution_percentage = 100

    scene.cycles.device = "GPU"
    scene.cycles.samples = 4096
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 4
    scene.cycles.transparent_max_bounces = 8
    scene.cycles.transmission_bounces = 12
    scene.cycles.filter_width = 1.5
    #scene.cycles.use_denoising = True

    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    # Set the device_type
    bpy.context.preferences.addons[
        "cycles"
    ].preferences.compute_device_type = "CUDA" # or "OPENCL"

    print(f"Rendering '{args.object_name}/model.obj' to '{target_dir}'")
    bpy.ops.wm.obj_import(filepath=obj_path)
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME')
    model = bpy.data.objects["model"]
    model.rotation_mode = 'XYZ'

    rotate_by_elev = 5   #How many degrees to rotate the knob for every step
    rotate_by_azi = 5   #How many degrees to rotate the knob for every step
    start_angle = 0      #What angle to start from
    for elev in range(60, 121, rotate_by_elev): # elev
        for azi in range(0, 361, rotate_by_azi): # azi
            if azi > 45 and azi < 315: continue
            rand_radius = np.random.rand(5) * 1.15 - 0.15 # -0.15 to 1.0
            for radius in list(rand_radius): # radius
                radius = np.round(radius, 2)
                render.filepath = f"{target_dir}/{90-elev}_{azi}_{radius:.2f}.png"
                if(os.path.exists(render.filepath)):
                    continue
                model.rotation_euler = (np.deg2rad(elev), np.deg2rad(azi), 0)
                model.location = (0, 0, 0)
                cam.location[2] = -0.35 - radius
                bpy.ops.render.render(write_still=True, use_viewport= True)
