import torch
import numpy as np
from plyfile import PlyData, PlyElement
from .general_utils import inverse_sigmoid, strip_symmetric, build_scaling_rotation
import utils3d
import todos
import pdb

class Gaussian:
    def __init__(
            self, 
            aabb : list,
            sh_degree : int = 0,
            mininum_kernel_size : float = 0.0,
            scaling_bias : float = 0.01,
            opacity_bias : float = 0.1,
            scaling_activation : str = "exp",
            device='cuda'
        ):
        # aabb = [-0.5, -0.5, -0.5, 1.0, 1.0, 1.0]
        # sh_degree = 0
        # mininum_kernel_size = 0.0009
        # scaling_bias = 0.004
        # opacity_bias = 0.1
        # scaling_activation = 'softplus'
        # device = 'cuda'

        self.init_params = {
            'aabb': aabb,
            'sh_degree': sh_degree,
            'mininum_kernel_size': mininum_kernel_size,
            'scaling_bias': scaling_bias,
            'opacity_bias': opacity_bias,
            'scaling_activation': scaling_activation,
        }
        
        self.sh_degree = sh_degree
        self.active_sh_degree = sh_degree
        self.mininum_kernel_size = mininum_kernel_size 
        self.scaling_bias = scaling_bias
        self.opacity_bias = opacity_bias
        self.scaling_activation_type = scaling_activation
        self.device = device
        self.aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
        self.setup_functions()

        self._xyz = None
        self._features_dc = None
        self._features_rest = None
        self._scaling = None
        self._rotation = None
        self._opacity = None


    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        if self.scaling_activation_type == "exp":
            pdb.set_trace()
            self.scaling_activation = torch.exp
            self.inverse_scaling_activation = torch.log
        elif self.scaling_activation_type == "softplus": # True
            self.scaling_activation = torch.nn.functional.softplus
            self.inverse_scaling_activation = lambda x: x + torch.log(-torch.expm1(-x))

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        
        self.scale_bias = self.inverse_scaling_activation(torch.tensor(self.scaling_bias)).cuda()
        # self.scale_bias -- tensor(-5.519460, device='cuda:0')
        self.rots_bias = torch.zeros((4)).cuda()
        self.rots_bias[0] = 1
        self.opacity_bias = self.inverse_opacity_activation(torch.tensor(self.opacity_bias)).cuda()
        # self.opacity_bias -- tensor(-2.197225, device='cuda:0')


    @property
    def get_scaling(self):
        # ==> pdb.set_trace()
        scales = self.scaling_activation(self._scaling + self.scale_bias)
        scales = torch.square(scales) + self.mininum_kernel_size ** 2
        scales = torch.sqrt(scales)
        return scales
    
    @property
    def get_rotation(self):
        # ==> pdb.set_trace()
        return self.rotation_activation(self._rotation + self.rots_bias[None, :])
    
    @property
    def get_xyz(self):
        # ==> pdb.set_trace()
        return self._xyz * self.aabb[None, 3:] + self.aabb[None, :3]
    
    @property
    def get_features(self):
        # ==> pdb.set_trace()
        assert self._features_rest == None
        return torch.cat((self._features_dc, self._features_rest), dim=2) if self._features_rest is not None else self._features_dc
    
    @property
    def get_opacity(self):
        # ==> pdb.set_trace()
        return self.opacity_activation(self._opacity + self.opacity_bias)
    
    def get_covariance(self, scaling_modifier = 1):
        pdb.set_trace()
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation + self.rots_bias[None, :])
    
