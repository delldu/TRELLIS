"""Image 3D Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2025(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, Wed 09 Apr 2025 10:36:34 AM CST
# ***
# ************************************************************************************/
#
__version__ = "1.0.0"

import os
from tqdm import tqdm
import torch
import torch.nn.functional as F

import numpy as np
from skimage import measure
import trimesh

import todos
import pdb

def image_center(image, border_ratio = 0.15):
    B, C, H, W = image.size()
    assert C == 4

    # 1) ============================================================================
    mask = image[:, 3:4, :, :]
    coords = torch.nonzero(mask.view(H, W), as_tuple=True)
    x1, y1, x2, y2 = coords[1].min(), coords[0].min(), coords[1].max(), coords[0].max()
    h = y2 - y1
    w = x2 - x1
    if h == 0 or w == 0:
        raise ValueError('input image is empty')
    scale = max(H, W) * (1.0 - border_ratio) / max(h, w) # 0.9550561797752809
    h2 = int(h * scale)
    w2 = int(w * scale)
    x2_min = (max(H, W) - w2) // 2
    x2_max = x2_min + w2
    y2_min = (max(H, W) - h2) // 2
    y2_max = y2_min + h2

    # 2) ============================================================================
    crop_image = image[:, :, y1:y2, x1:x2]
    new_image = torch.zeros(B, C, max(H, W), max(H, W))
    new_image[:, :, y2_min:y2_max, x2_min:x2_max] = \
        F.interpolate(crop_image, size=(h2, w2), mode="bilinear", align_corners=True)

    new_bg = torch.ones(B, 3, max(H, W), max(H, W))
    new_mask = new_image[:, 3:4, :, :]
    new_image = new_image[:, 0:3, :, :] * new_mask + new_bg * (1.0 - new_mask)

    return new_image, new_mask

def create_mesh(grid_logit):
    '''Create mesh from grid logit'''

    mesh_v, mesh_f, normals, _ = measure.marching_cubes(
        grid_logit.cpu().numpy(),
        0.0, # mc_level
        method="lewiner"
    )
    # array [mesh_v] shape: (327988, 3), min: 1.8486219644546509, max: 382.1461181640625, mean: 184.00257873535156
    # array [mesh_f] shape: (655980, 3), min: 0, max: 327987, mean: 163994.911803
    # array [normals] shape: (327988, 3), min: -1.0, max: 1.0, mean: 0.005313000176101923
    grid_size = [385, 385, 385]
    bbox_min = np.array([-1.01, -1.01, -1.01])
    bbox_size = np.array([2.02,  2.02,  2.02])
    mesh_v = mesh_v / grid_size * bbox_size + bbox_min

    mesh_f = mesh_f[:, ::-1] # !!!! [0, 1, 2] ==> [2, 1, 0] !!!
    mesh = trimesh.Trimesh(mesh_v, mesh_f)
    # mesh ...
    return mesh # mesh.export("xxxx.glb")


def get_shape_model():
    """Create model."""

    # model = vae.ShapeVAE()
    # model = dit.Hunyuan3DDiT()
    device = todos.model.get_device()    
    model = shape.ShapeGenerator(device)
    # model = model.to(device)
    model.eval()

    if "cpu" in str(device.type):
        model.float()

    print(f"Running on {device} ...")
    # # make sure model good for C/C++
    # model = torch.jit.script(model)
    # # https://github.com/pytorch/pytorch/issues/52286
    # torch._C._jit_set_profiling_executor(False)
    # # C++ Reference
    # # torch::jit::getProfilingMode() = false;
    # # torch::jit::setTensorExprFuserEnabled(false);

    # todos.data.mkdir("output")
    # if not os.path.exists("output/image3d.torch"):
    #     model.save("output/image3d.torch")
    # torch.save(model.state_dict(), "/tmp/image3d.pth")

    return model, device


def predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    model, device = get_shape_model()

    # load files
    input_filenames = todos.data.load_files(input_files)

    # start predict
    for filename in input_filenames:
        print(f"Create mesh from {filename} ...")

        input_tensor = todos.data.load_rgba_tensor(filename)
        input_image, input_mask = image_center(input_tensor)
        input_image = input_image.to(device)

        # model = model.half()
        with torch.no_grad():
            grid_logits = model(input_image)

        # output_file = f"{output_dir}/{os.path.basename(filename)}"
        obj_filename = os.path.basename(filename)
        obj_filename = obj_filename.replace(".jpg", ".obj")
        obj_filename = obj_filename.replace(".png", ".obj")
        output_file = f"{output_dir}/{obj_filename}"
        output_mesh = create_mesh(grid_logits[0])
        output_mesh.export(output_file)
