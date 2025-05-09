from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...modules import sparse as sp
from ...utils.random_utils import hammersley_sequence
from .base import SparseTransformerBase
from ...representations import Gaussian
import pdb

class SLatGaussianDecoder(SparseTransformerBase):
    def __init__(
        self,
        resolution: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict = None,
    ):
        super().__init__(
            in_channels=latent_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            qk_rms_norm=qk_rms_norm,
        )
        # print(f"SLatGaussianDecoder: resolution={resolution}, model_channels={model_channels}, latent_channels={latent_channels}, num_blocks={num_blocks}, num_heads={num_heads}, attn_mode={attn_mode}, window_size={window_size}, qk_rms_norm={qk_rms_norm}")
        # SLatGaussianDecoder: resolution=64, model_channels=768, latent_channels=8, num_blocks=12, num_heads=12, attn_mode=swin, window_size=8, qk_rms_norm=False

        assert resolution == 64
        assert model_channels == 768
        assert latent_channels == 8
        assert num_blocks == 12
        assert num_heads == 12
        assert num_head_channels == 64
        assert mlp_ratio == 4
        assert attn_mode == 'swin'
        assert window_size == 8
        assert pe_mode == 'ape'
        assert use_fp16 == True
        assert qk_rms_norm == False
        
        # representation_config = {'lr': {'_xyz': 1.0, '_features_dc': 1.0, 
        #     '_opacity': 1.0, '_scaling': 1.0, '_rotation': 0.1}, 
        #     'perturb_offset': True, 'voxel_size': 1.5, 'num_gaussians': 32, '2d_filter_kernel_size': 0.1, 
        #     '3d_filter_kernel_size': 0.0009, 'scaling_bias': 0.004, 'opacity_bias': 0.1, 'scaling_activation': 'softplus'}

        self.resolution = resolution
        self.rep_config = representation_config
        self._calc_layout()
        self.out_layer = sp.SparseLinear(model_channels, self.out_channels)
        self._build_perturbation()

        if use_fp16:
            self.convert_to_fp16()


    def _build_perturbation(self) -> None:
        perturbation = [hammersley_sequence(3, i, self.rep_config['num_gaussians']) for i in range(self.rep_config['num_gaussians'])]
        perturbation = torch.tensor(perturbation).float() * 2 - 1
        perturbation = perturbation / self.rep_config['voxel_size']
        perturbation = torch.atanh(perturbation).to(self.device)
        self.register_buffer('offset_perturbation', perturbation)

    def _calc_layout(self) -> None:
        self.layout = {
            '_xyz' : {'shape': (self.rep_config['num_gaussians'], 3), 'size': self.rep_config['num_gaussians'] * 3},
            '_features_dc' : {'shape': (self.rep_config['num_gaussians'], 1, 3), 'size': self.rep_config['num_gaussians'] * 3},
            '_scaling' : {'shape': (self.rep_config['num_gaussians'], 3), 'size': self.rep_config['num_gaussians'] * 3},
            '_rotation' : {'shape': (self.rep_config['num_gaussians'], 4), 'size': self.rep_config['num_gaussians'] * 4},
            '_opacity' : {'shape': (self.rep_config['num_gaussians'], 1), 'size': self.rep_config['num_gaussians']},
        }
        start = 0
        for k, v in self.layout.items():
            v['range'] = (start, start + v['size'])
            start += v['size']
        self.out_channels = start
    
    def to_representation(self, x: sp.SparseTensor) -> List[Gaussian]:
        """
        Convert a batch of network outputs to 3D representations.

        Args:
            x: The [N x * x C] sparse tensor output by the network.

        Returns:
            list of representations
        """
        ret = []
        for i in range(x.shape[0]):
            representation = Gaussian(
                sh_degree=0,
                aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
                mininum_kernel_size = self.rep_config['3d_filter_kernel_size'],
                scaling_bias = self.rep_config['scaling_bias'],
                opacity_bias = self.rep_config['opacity_bias'],
                scaling_activation = self.rep_config['scaling_activation']
            )
            xyz = (x.coords[x.layout[i]][:, 1:].float() + 0.5) / self.resolution
            for k, v in self.layout.items():
                if k == '_xyz':
                    offset = x.feats[x.layout[i]][:, v['range'][0]:v['range'][1]].reshape(-1, *v['shape'])
                    offset = offset * self.rep_config['lr'][k]
                    if self.rep_config['perturb_offset']:
                        offset = offset + self.offset_perturbation
                    offset = torch.tanh(offset) / self.resolution * 0.5 * self.rep_config['voxel_size']
                    _xyz = xyz.unsqueeze(1) + offset
                    setattr(representation, k, _xyz.flatten(0, 1))
                else:
                    feats = x.feats[x.layout[i]][:, v['range'][0]:v['range'][1]].reshape(-1, *v['shape']).flatten(0, 1)
                    feats = feats * self.rep_config['lr'][k]
                    setattr(representation, k, feats)
            ret.append(representation)
        return ret

    def forward(self, x: sp.SparseTensor) -> List[Gaussian]:
        # tensor [x data.coords] size: [14955, 4], min: 0.0, max: 63.0, mean: 23.262018
        # tensor [x data.features] size: [14955, 8], min: -9.592283, max: 9.934357, mean: -0.068937

        h2 = super().forward(x)
        h2 = h2.type(x.dtype)
        h2 = h2.replace(F.layer_norm(h2.feats, h2.feats.shape[-1:]))
        h2 = self.out_layer.float()(h2)
        h2 = self.to_representation(h2)

        # h2 -- [<trellis.representations.gaussian.gaussian_model.Gaussian object at 0x7f6db8dae310>]
        # h2[0].get_xyz.size() -- [478560, 3]
        # h2[0].get_features.size() -- [478560, 1, 3]
        # h2[0].get_scaling.size() -- [478560, 3]
        # h2[0].get_rotation.size() -- [478560, 4]
        # h2[0].sh_degree === 0
        # (Pdb) h2[0].aabb.size() -- [6]
        # (Pdb) h2[0].aabb -- tensor([-0.500000, -0.500000, -0.500000,  1.000000,  1.000000,  1.000000], device='cuda:0')

        return h2
    
