#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from easydict import EasyDict as edict
import numpy as np
from ..representations.gaussian import Gaussian
# from .sh_utils import eval_sh
import torch.nn.functional as F
# from easydict import EasyDict as edict
import todos
import pdb

def intrinsics_to_projection(
        intrinsics: torch.Tensor,
        near: float,
        far: float,
    ) -> torch.Tensor:
    """
    OpenCV intrinsics to OpenGL perspective matrix

    Args:
        intrinsics (torch.Tensor): [3, 3] OpenCV intrinsics matrix
        near (float): near plane to clip
        far (float): far plane to clip
    Returns:
        (torch.Tensor): [4, 4] OpenGL perspective matrix
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = - 2 * cy + 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.
    return ret

# gaussian_render
def gs_render(viewpoint_camera, pc : Gaussian, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # lazy import
    # ------------------------------------------------------------------------------------------------------------
    # viewpoint_camera.keys() -- ['image_height', 'image_width', 'FoVx', 'FoVy', 'znear', 'zfar', 
    #     'world_view_transform', 'projection_matrix', 'full_proj_transform', 'camera_center']
    # ------------------------------------------------------------------------------------------------------------

    if 'GaussianRasterizer' not in globals():
        from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    kernel_size = pipe.kernel_size # 0.1
    subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2), dtype=torch.float32, device="cuda")


    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height), # 1024
        image_width=int(viewpoint_camera.image_width), # 1024
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        # kernel_size=kernel_size,
        # subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python: # False
        pdb.set_trace()

        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        # tensor [pc.get_scaling] size: [478560, 3], min: 0.0009, max: 0.020283, mean: 0.001552
        # tensor [pc.get_rotation] size: [478560, 4], min: -0.999981, max: 0.999999, mean: 0.206155


    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None: # True
        if pipe.convert_SHs_python: # False
            pdb.set_trace()
            # shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            # dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            # dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            # sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            # colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        pdb.set_trace()
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D, # size() -- [478560, 3]
        means2D = means2D, # size() -- [478560, 3]
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp # None ???
    )

    # tensor [rendered_image] size: [3, 1024, 1024], min: 0.0, max: 0.733704, mean: 0.004028
    # tensor [radii] size: [478560], min: 3.0, max: 35.0, mean: 5.125673
    # tensor [screenspace_points] size: [478560, 3], min: 0.0, max: 0.0, mean: 0.0

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # return edict({"render": rendered_image,
    #         "viewspace_points": screenspace_points,
    #         "visibility_filter" : radii > 0,
    #         "radii": radii})
    return edict({"render": rendered_image})

class GaussianRenderer:
    """
    Renderer for the Voxel representation.

    Args:
        rendering_options (dict): Rendering options.
    """

    def __init__(self, rendering_options={}) -> None:
        self.pipe = edict({
            "kernel_size": 0.1,
            "convert_SHs_python": False,
            "compute_cov3D_python": False,
            "scale_modifier": 1.0,
            "debug": False
        })
        self.rendering_options = edict({
            "resolution": None,
            "near": None,
            "far": None,
            "ssaa": 1,
            "bg_color": 'random',
        })
        self.rendering_options.update(rendering_options)
        self.bg_color = None
        # self.pipe -- {'kernel_size': 0.1, 'convert_SHs_python': False, 'compute_cov3D_python': False, 'scale_modifier': 1.0, 'debug': False}
        # self.rendering_options -- {'resolution': None, 'near': None, 'far': None, 'ssaa': 1, 'bg_color': 'random'}

    def render(
            self,
            gausssian: Gaussian,
            extrinsics: torch.Tensor,
            intrinsics: torch.Tensor,
            colors_overwrite: torch.Tensor = None
        ) -> edict:
        """
        Render the gausssian.

        Args:
            gaussian : gaussianmodule
            extrinsics (torch.Tensor): (4, 4) camera extrinsics
            intrinsics (torch.Tensor): (3, 3) camera intrinsics
            colors_overwrite (torch.Tensor): (N, 3) override color

        Returns:
            edict containing:
                color (torch.Tensor): (3, H, W) rendered color image
        """
        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]
        # self.rendering_options -- {'resolution': 1024, 'near': 0.8, 'far': 1.6, 'ssaa': 1, 'bg_color': [0, 0, 0]}

        if self.rendering_options["bg_color"] == 'random': # False
            pdb.set_trace()
            self.bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
            if np.random.rand() < 0.5:
                self.bg_color += 1
        else:
            self.bg_color = torch.tensor(self.rendering_options["bg_color"], dtype=torch.float32, device="cuda")

        view = extrinsics
        perspective = intrinsics_to_projection(intrinsics, near, far)
        camera = torch.inverse(view)[:3, 3]
        focalx = intrinsics[0, 0]
        focaly = intrinsics[1, 1]
        fovx = 2 * torch.atan(0.5 / focalx)
        fovy = 2 * torch.atan(0.5 / focaly)
            
        camera_dict = edict({
            "image_height": resolution * ssaa,
            "image_width": resolution * ssaa,
            "FoVx": fovx,
            "FoVy": fovy,
            "znear": near,
            "zfar": far,
            "world_view_transform": view.T.contiguous(),
            "projection_matrix": perspective.T.contiguous(),
            "full_proj_transform": (perspective @ view).T.contiguous(),
            "camera_center": camera
        })
        # ---------------------------------------------------------------------------------------------
        # (Pdb) for k, v in camera_dict.items(): print(k, v)
        # image_height 1024
        # image_width 1024
        # FoVx tensor(0.698132, device='cuda:0')
        # FoVy tensor(0.698132, device='cuda:0')
        # znear 0.8
        # zfar 1.6
        # world_view_transform tensor([[     1.000000,      0.000000,      0.000000,      0.000000],
        #         [     0.000000,      1.000000,      0.000000,      0.000000],
        #         [     0.000000,     -0.000000,      1.000000,      0.000000],
        #         [    -0.000000,     -0.000000,      2.000000,      1.000000]],
        #        device='cuda:0')
        # projection_matrix tensor([[ 2.747478,  0.000000,  0.000000,  0.000000],
        #         [ 0.000000,  2.747478,  0.000000,  0.000000],
        #         [ 0.000000,  0.000000,  2.000000,  1.000000],
        #         [ 0.000000,  0.000000, -1.600000,  0.000000]], device='cuda:0')
        # full_proj_transform tensor([[     2.747478,      0.000000,      0.000000,      0.000000],
        #         [     0.000000,      2.747478,      0.000000,      0.000000],
        #         [     0.000000,     -0.000000,      2.000000,      1.000000],
        #         [     0.000000,      0.000000,      2.400000,      2.000000]],
        #        device='cuda:0')
        # camera_center tensor([     0.000000,     -0.000000,     -2.000000], device='cuda:0')
        # ---------------------------------------------------------------------------------------------

        # Render
        assert colors_overwrite == None
        # self.bg_color -- tensor([0., 0., 0.], device='cuda:0')
        # self.pipe -- {'kernel_size': 0.1, 'convert_SHs_python': False, 'compute_cov3D_python': False, 'scale_modifier': 1.0, 'debug': False, 'use_mip_gaussian': True}

        render_ret = gs_render(camera_dict, gausssian, self.pipe, self.bg_color, override_color=colors_overwrite, 
            scaling_modifier=self.pipe.scale_modifier) # self.pipe.scale_modifier -- 1.0

        assert ssaa == 1
        if ssaa > 1: # False
            pdb.set_trace()
            render_ret.render = F.interpolate(render_ret.render[None], size=(resolution, resolution), mode='bilinear', align_corners=False, antialias=True).squeeze()
            
        ret = edict({
            'color': render_ret['render'] # render_ret['render'].size() -- [3, 1024, 1024]

        })
        return ret
