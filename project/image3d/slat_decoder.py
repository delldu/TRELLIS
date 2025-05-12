import os
from typing import *
import torch
import torch.nn as nn
from sparse_tensor import (
    SparseTensor, 
    SparseLinear, 
    SparseMultiHeadAttention, 
    SparseFeedForwardNet, 
    SparseGroupNorm32,
    SparseSiLU, 
    SparseGELU, 
    SparseSubdivide, 
    SparseConv3d,
    )
from abs_position_embed import AbsolutePositionEmbedder
from sparse_structure_decoder import LayerNorm32
from gaussian_render import Gaussian
from render_utils import hammersley_sequence
from mesh_flexicubes import FlexiCubes
from easydict import EasyDict as edict

import pdb

class SparseTransformerBlock(nn.Module):
    """
    Sparse Transformer block (MSA + FFN).
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode = None,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        ln_affine: bool = False,
    ):
        super().__init__()
        assert shift_sequence == None
        assert serialize_mode == None

        self.norm1 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_sequence=shift_sequence,
            shift_window=shift_window,
            serialize_mode=serialize_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )

    def _forward(self, x: SparseTensor) -> SparseTensor:
        h = x.replace(self.norm1.float()(x.feats))
        h = self.attn(h)
        x = x + h
        h = x.replace(self.norm2.float()(x.feats))
        h = self.mlp(h)
        x = x + h
        return x

    def forward(self, x: SparseTensor) -> SparseTensor:
        return self._forward(x)




def block_attn_config(self):
    """
    Return the attention configuration of the model.
    """
    for i in range(self.num_blocks):
        if self.attn_mode == "shift_window":
            yield "serialized", self.window_size, 0, (16 * (i % 2),) * 3, sp.SerializeMode.Z_ORDER
        elif self.attn_mode == "shift_sequence":
            yield "serialized", self.window_size, self.window_size // 2 * (i % 2), (0, 0, 0), sp.SerializeMode.Z_ORDER
        elif self.attn_mode == "shift_order":
            yield "serialized", self.window_size, 0, (0, 0, 0), sp.SerializeModes[i % 4]
        elif self.attn_mode == "full":
            yield "full", None, None, None, None
        elif self.attn_mode == "swin":
            # attn_mode, window_size, shift_sequence, shift_window, serialize_mode
            yield "windowed", self.window_size, None, self.window_size // 2 * (i % 2), None


class SparseTransformerBase(nn.Module):
    """
    Sparse Transformer without output layers.
    Serve as the base class for encoder and decoder.
    """
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        # print(f"== SparseTransformerBase: attn_mode={attn_mode}, window_size={window_size}")
        # == SparseTransformerBase: attn_mode=swin, window_size=8

        # assert in_channels == 8
        # assert model_channels == 768
        # assert num_blocks == 12
        # assert num_heads == 12
        assert num_head_channels == 64
        assert mlp_ratio == 4
        assert attn_mode == 'swin'
        assert window_size == 8
        assert pe_mode == 'ape'
        # assert qk_rms_norm == False

        self.num_blocks = num_blocks
        self.window_size = window_size
        self.num_heads = num_heads or model_channels // num_head_channels
        self.attn_mode = attn_mode # 'swin'
        self.pe_mode = pe_mode # "ape"
        self.dtype = torch.float16

        if pe_mode == "ape": # True
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.input_layer = SparseLinear(in_channels, model_channels)
        self.blocks = nn.ModuleList([
            SparseTransformerBlock(
                model_channels,
                num_heads=self.num_heads,
                mlp_ratio=mlp_ratio,
                attn_mode=attn_mode,
                window_size=window_size,
                shift_sequence=shift_sequence,
                shift_window=shift_window,
                serialize_mode=serialize_mode,
                use_rope=(pe_mode == "rope"),
                qk_rms_norm=qk_rms_norm,
            )
            for attn_mode, window_size, shift_sequence, shift_window, serialize_mode in block_attn_config(self)
        ])

    def forward(self, x: SparseTensor) -> SparseTensor:
        # ==> pdb.set_trace()
        h2 = self.input_layer.float()(x)
        if self.pe_mode == "ape": # True
            h2 = h2 + self.pos_embedder(x.coords[:, 1:])
        h2 = h2.type(self.dtype)
        for block in self.blocks:
            h2 = block(h2)

        return h2

# xxxx_1111
class SLatGaussianDecoder(SparseTransformerBase):
    def __init__(
        self,
        resolution = 64,
        model_channels = 768,
        latent_channels =8,
        num_blocks = 12,
        num_heads = 12,
        num_head_channels = 64,
        mlp_ratio: float = 4,
        attn_mode = "swin",
        window_size = 8,
        pe_mode = "ape",
        qk_rms_norm = False,
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
        assert qk_rms_norm == False
        representation_config = {'lr': {'_xyz': 1.0, '_features_dc': 1.0, 
            '_opacity': 1.0, '_scaling': 1.0, '_rotation': 0.1}, 
            'perturb_offset': True, 'voxel_size': 1.5, 'num_gaussians': 32, '2d_filter_kernel_size': 0.1, 
            '3d_filter_kernel_size': 0.0009, 'scaling_bias': 0.004, 'opacity_bias': 0.1, 'scaling_activation': 'softplus'}

        self.resolution = resolution
        self.rep_config = representation_config
        self._calc_layout()
        self.out_layer = SparseLinear(model_channels, self.out_channels)
        self._build_perturbation()


    def load_weights(self, model_path="models/image3d_dinov2.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        print(f"Loading {checkpoint} ...")
        self.load_state_dict(torch.load(checkpoint), strict=True)


    def _build_perturbation(self) -> None:
        perturbation = [hammersley_sequence(3, i, self.rep_config['num_gaussians']) for i in range(self.rep_config['num_gaussians'])]
        perturbation = torch.tensor(perturbation).float() * 2 - 1
        perturbation = perturbation / self.rep_config['voxel_size']
        perturbation = torch.atanh(perturbation)
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

    def to_representation(self, x: SparseTensor) -> List[Gaussian]:
        """
        Convert a batch of network outputs to 3D representations.

        Args:
            x: The [N x * x C] sparse tensor output by the network.

        Returns:
            list of representations
        """
        ret = []
        for i in range(x.shape[0]): # 1
            # xxxx_3333
            representation = Gaussian(
                sh_degree=0,
                aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
                mininum_kernel_size = self.rep_config['3d_filter_kernel_size'],
                scaling_bias = self.rep_config['scaling_bias'],
                opacity_bias = self.rep_config['opacity_bias'],
                scaling_activation = self.rep_config['scaling_activation']
            )
            # self.resolution === 64
            # x.layout[i] -- slice(0, 14955, None)

            # tensor [x.coords] size: [14955, 4], min: 0.0, max: 63.0, mean: 23.262018
            xyz = (x.coords[x.layout[i]][:, 1:].float() + 0.5) / self.resolution
            # tensor [xyz] size: [14955, 3], min: 0.007812, max: 0.992188, mean: 0.492438

            # (Pdb) self.layout.keys() -- ['_xyz', '_features_dc', '_scaling', '_rotation', '_opacity']
            for k, v in self.layout.items():
                if k == '_xyz':
                    offset = x.feats[x.layout[i]][:, v['range'][0]:v['range'][1]].reshape(-1, *v['shape'])
                    offset = offset * self.rep_config['lr'][k]
                    if self.rep_config['perturb_offset']: # True
                        offset = offset + self.offset_perturbation
                    offset = torch.tanh(offset) / self.resolution * 0.5 * self.rep_config['voxel_size']
                    _xyz = xyz.unsqueeze(1) + offset
                    setattr(representation, k, _xyz.flatten(0, 1))
                else:
                    feats = x.feats[x.layout[i]][:, v['range'][0]:v['range'][1]].reshape(-1, *v['shape']).flatten(0, 1)
                    # (Pdb) self.rep_config['lr'] -- {'_xyz': 1.0, '_features_dc': 1.0, '_opacity': 1.0, '_scaling': 1.0, '_rotation': 0.1}
                    feats = feats * self.rep_config['lr'][k]
                    setattr(representation, k, feats)
            ret.append(representation)
            
        return ret

    def forward(self, x: SparseTensor) -> List[Gaussian]:
        # tensor [x data.coords] size: [14955, 4], min: 0.0, max: 63.0, mean: 23.262018
        # tensor [x data.features] size: [14955, 8], min: -9.592283, max: 9.934357, mean: -0.068937

        h2 = super().forward(x)
        h2 = h2.type(x.dtype)
        h2 = h2.replace(F.layer_norm(h2.feats, h2.feats.shape[-1:]))
        h2 = self.out_layer.float()(h2)
        h2 = self.to_representation(h2)

        # h2 -- [Gaussian object]
        # h2[0].get_xyz.size() -- [478560, 3]
        # h2[0].get_features.size() -- [478560, 1, 3]
        # h2[0].get_scaling.size() -- [478560, 3]
        # h2[0].get_rotation.size() -- [478560, 4]
        # h2[0].sh_degree === 0
        # (Pdb) h2[0].aabb.size() -- [6]
        # (Pdb) h2[0].aabb -- tensor([-0.500000, -0.500000, -0.500000,  1.000000,  1.000000,  1.000000], device='cuda:0')

        return h2
    
class SparseSubdivideBlock3d(nn.Module):
    """
    A 3D subdivide block that can subdivide the sparse tensor.

    Args:
        channels: channels in the inputs and outputs.
        out_channels: if specified, the number of output channels.
        num_groups: the number of groups for the group norm.
    """
    def __init__(
        self,
        channels: int,
        resolution: int,
        out_channels: Optional[int] = None,
        num_groups: int = 32
    ):
        super().__init__()
        # channels = 768
        # resolution = 64
        # out_channels = 192
        # num_groups = 32

        # self.channels = channels
        # self.resolution = resolution
        self.out_resolution = resolution * 2
        self.out_channels = out_channels or channels

        self.act_layers = nn.Sequential(
            SparseGroupNorm32(num_groups, channels), # (32, 768)
            SparseSiLU()
        )
        
        self.sub = SparseSubdivide()
        
        self.out_layers = nn.Sequential(
            SparseConv3d(channels, self.out_channels, 3, indice_key=f"res_{self.out_resolution}"),
            SparseGroupNorm32(num_groups, self.out_channels),
            SparseSiLU(),
            SparseConv3d(self.out_channels, self.out_channels, 3, indice_key=f"res_{self.out_resolution}"),
        )
        
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = SparseConv3d(channels, self.out_channels, 1, indice_key=f"res_{self.out_resolution}")
        
    def forward(self, x: SparseTensor) -> SparseTensor:
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        Args:
            x: an [N x C x ...] Tensor of features.
        Returns:
            an [N x C x ...] Tensor of outputs.
        """

        h2 = self.act_layers.float()(x) # SparseGroupNorm32
        h2 = self.sub(h2)
        x = self.sub(x)
        h2 = self.out_layers.float()(h2.float())
        h2 = h2 + self.skip_connection(x)

        return h2

# xxxx_1111
class SLatMeshDecoder(SparseTransformerBase):
    def __init__(
        self,
        resolution = 64,
        model_channels = 768,
        latent_channels = 8,
        num_blocks = 12,
        num_heads = 12,
        num_head_channels = 64,
        mlp_ratio = 4,
        attn_mode = "swin",
        window_size = 8,
        pe_mode = "ape",
        qk_rms_norm = False,
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
            qk_rms_norm=qk_rms_norm,
        )
        # print(f"SLatMeshDecoder: resolution={resolution}, model_channels={model_channels}, latent_channels={latent_channels}, num_blocks={num_blocks}, num_heads={num_heads}, window_size={window_size}, pe_mode={pe_mode}, qk_rms_norm={qk_rms_norm}, representation_config={representation_config}")
        # SLatMeshDecoder: resolution=64, model_channels=768, latent_channels=8, num_blocks=12, num_heads=12, window_size=8, 
        #pe_mode=ape, qk_rms_norm=False, representation_config={'use_color': True}

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
        assert qk_rms_norm == False
        representation_config = {'use_color': True}
        
        self.rep_config = representation_config

        # xxxx_3333
        self.mesh_extractor = SparseFeatures2Mesh(res=resolution*4, use_color=self.rep_config.get('use_color', False))
        self.out_channels = self.mesh_extractor.feats_channels

        self.upsample = nn.ModuleList([
            SparseSubdivideBlock3d(
                channels=model_channels, # 768
                resolution=resolution, # 64
                out_channels=model_channels // 4 # 192
            ),
            SparseSubdivideBlock3d(
                channels=model_channels // 4, # 192 ???
                resolution=resolution * 2, # 128 ???
                out_channels=model_channels // 8 # 96 ???
            )
        ])
        self.out_layer = SparseLinear(model_channels // 8, self.out_channels)

    def to_representation(self, x: SparseTensor):
        """
        Convert a batch of network outputs to 3D representations.

        Args:
            x: The [N x * x C] sparse tensor output by the network.

        Returns:
            list of representations
        """
        ret = []
        for i in range(x.shape[0]):
            mesh = self.mesh_extractor(x[i], training=self.training)
            ret.append(mesh)
        return ret

    def forward(self, x: SparseTensor):
        # tensor [x data.coords] size: [14955, 4], min: 0.0, max: 63.0, mean: 23.262018
        # tensor [x data.features] size: [14955, 8], min: -9.592283, max: 9.934357, mean: -0.068937

        h2 = super().forward(x)

        for block in self.upsample: # SparseSubdivideBlock3d
            h2 = block.float()(h2.float())

        h2 = h2.type(x.dtype)

        h2 = self.out_layer.float()(h2)

        h2 = self.to_representation(h2)

        # h2 -- [MeshExtractResult object>]
        # tensor [h2[0].vertices] size: [298216, 3], min: -0.500411, max: 0.49844, mean: -0.0098
        # tensor [h2[0].faces] size: [596762, 3], min: 0.0, max: 298215.0, mean: 148879.0
        # tensor [h2[0].vertex_attrs] size: [298216, 6], min: 2.9e-05, max: 0.999892, mean: 0.34404
        # tensor [h2[0].face_normal] size: [596762, 3, 3], min: -1.0, max: 1.0, mean: 0.012801
        # h2[0].res === 256
        # h2[0].success === True
        return h2
    
cube_corners = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [
        1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.int)


# --------------------------------------------------------------        
def construct_dense_grid(res, device='cuda'):
    '''construct a dense grid based on resolution'''
    res_v = res + 1
    vertsid = torch.arange(res_v ** 3, device=device)
    coordsid = vertsid.reshape(res_v, res_v, res_v)[:res, :res, :res].flatten()
    cube_corners_bias = (cube_corners[:, 0] * res_v + cube_corners[:, 1]) * res_v + cube_corners[:, 2]
    cube_fx8 = (coordsid.unsqueeze(1) + cube_corners_bias.unsqueeze(0).to(device))
    verts = torch.stack([vertsid // (res_v ** 2), (vertsid // res_v) % res_v, vertsid % res_v], dim=1)
    return verts, cube_fx8

# --------------------------------------------------------------        
def construct_voxel_grid(coords):
    verts = (cube_corners.unsqueeze(0).to(coords) + coords.unsqueeze(1)).reshape(-1, 3)
    verts_unique, inverse_indices = torch.unique(verts, dim=0, return_inverse=True)
    cubes = inverse_indices.reshape(-1, 8)
    return verts_unique, cubes

# --------------------------------------------------------------        
def cubes_to_verts(num_verts, cubes, value, reduce='mean'):
    """
    Args:
        cubes [Vx8] verts index for each cube
        value [Vx8xM] value to be scattered
    Operation:
        reduced[cubes[i][j]][k] += value[i][k]
    """
    M = value.shape[2] # number of channels
    reduced = torch.zeros(num_verts, M, device=cubes.device)
    return torch.scatter_reduce(reduced, 0, 
        cubes.unsqueeze(-1).expand(-1, -1, M).flatten(0, 1), 
        value.flatten(0, 1), reduce=reduce, include_self=False)

# --------------------------------------------------------------        
def sparse_cube2verts(coords, feats, training=True):
    new_coords, cubes = construct_voxel_grid(coords)
    new_feats = cubes_to_verts(new_coords.shape[0], cubes, feats)
    assert training == False
    if training:
        con_loss = torch.mean((feats - new_feats[cubes]) ** 2)
    else:
        con_loss = 0.0
    return new_coords, new_feats, con_loss
    
# --------------------------------------------------------------        
def get_dense_attrs(coords : torch.Tensor, feats : torch.Tensor, res : int, sdf_init=True):
    F = feats.shape[-1]
    dense_attrs = torch.zeros([res] * 3 + [F], device=feats.device)
    assert sdf_init == False
    if sdf_init:
        dense_attrs[..., 0] = 1 # initial outside sdf value
    dense_attrs[coords[:, 0], coords[:, 1], coords[:, 2], :] = feats
    return dense_attrs.reshape(-1, F)

# --------------------------------------------------------------        
def get_defomed_verts(v_pos : torch.Tensor, deform : torch.Tensor, res):
    return v_pos / res - 0.5 + (1 - 1e-8) / (res * 2) * torch.tanh(deform)
        

# xxxx_3333
class MeshExtractResult:
    def __init__(self,
        vertices,
        faces,
        vertex_attrs=None,
        res=64
    ):
        self.vertices = vertices
        self.faces = faces.long()
        self.vertex_attrs = vertex_attrs
        self.face_normal = self.comput_face_normals(vertices, faces)
        self.res = res
        self.success = (vertices.shape[0] != 0 and faces.shape[0] != 0)

        # training only
        self.tsdf_v = None
        self.tsdf_s = None
        self.reg_loss = None
        
    def comput_face_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[i0, :]
        v1 = verts[i1, :]
        v2 = verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)
        return face_normals[:, None, :].repeat(1, 3, 1)
                
class SparseFeatures2Mesh:
    def __init__(self, device="cuda", res=64, use_color=True):
        '''
        a model to generate a mesh from sparse features structures using flexicube
        '''
        super().__init__()
        self.device=device
        self.res = res
        self.mesh_extractor = FlexiCubes(device=device)
        self.sdf_bias = -1.0 / res
        verts, cube = construct_dense_grid(self.res, self.device)
        self.reg_c = cube.to(self.device)
        self.reg_v = verts.to(self.device)
        self.use_color = use_color
        self._calc_layout()
        # ==> pdb.set_trace()
    
    def _calc_layout(self):
        LAYOUTS = {
            'sdf': {'shape': (8, 1), 'size': 8},
            'deform': {'shape': (8, 3), 'size': 8 * 3},
            'weights': {'shape': (21,), 'size': 21}
        }
        if self.use_color:
            '''
            6 channel color including normal map
            '''
            LAYOUTS['color'] = {'shape': (8, 6,), 'size': 8 * 6}
        self.layouts = edict(LAYOUTS)
        start = 0
        for k, v in self.layouts.items():
            v['range'] = (start, start + v['size'])
            start += v['size']
        self.feats_channels = start

    # --------------------------------------------------------------        
    def get_layout(self, feats : torch.Tensor, name : str):
        if name not in self.layouts:
            return None
        return feats[:, self.layouts[name]['range'][0]:self.layouts[name]['range'][1]].reshape(-1, *self.layouts[name]['shape'])
    
    def __call__(self, cubefeats : SparseTensor, training=False):
        """
        Generates a mesh based on the specified sparse voxel structures.
        Args:
            cube_attrs [Nx21] : Sparse Tensor attrs about cube weights
            verts_attrs [Nx10] : [0:1] SDF [1:4] deform [4:7] color [7:10] normal 
        Returns:
            return the success tag and ni you loss, 
        """
        # ---------------------------------------------------------------------------------------
        # cubefeats = <trellis.modules.sparse.basic.SparseTensor object at 0x7fcf9cd0b490>
        # training = False
        # ---------------------------------------------------------------------------------------

        # add sdf bias to verts_attrs
        coords = cubefeats.coords[:, 1:]
        feats = cubefeats.feats
        
        sdf, deform, color, weights = [self.get_layout(feats, name) for name in ['sdf', 'deform', 'color', 'weights']]
        sdf += self.sdf_bias
        v_attrs = [sdf, deform, color] if self.use_color else [sdf, deform]
        v_pos, v_attrs, reg_loss = sparse_cube2verts(coords, torch.cat(v_attrs, dim=-1), training=training)
        v_attrs_d = get_dense_attrs(v_pos, v_attrs, res=self.res+1, sdf_init=True)
        weights_d = get_dense_attrs(coords, weights, res=self.res, sdf_init=False)
        if self.use_color:
            sdf_d, deform_d, colors_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4], v_attrs_d[..., 4:]
        else:
            sdf_d, deform_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4]
            colors_d = None
            
        x_nx3 = get_defomed_verts(self.reg_v, deform_d, self.res)
        
        vertices, faces, L_dev, colors = self.mesh_extractor(
            voxelgrid_vertices=x_nx3,
            scalar_field=sdf_d,
            cube_idx=self.reg_c,
            resolution=self.res,
            beta=weights_d[:, :12],
            alpha=weights_d[:, 12:20],
            gamma_f=weights_d[:, 20],
            voxelgrid_colors=colors_d,
            training=training)
        
        # ===> pdb.set_trace()
        mesh = MeshExtractResult(vertices=vertices, faces=faces, vertex_attrs=colors, res=self.res)
        return mesh


if __name__ == "__main__":
    model = SLatGaussianDecoder()
    model.eval()
    print(model)

    print("-" * 80)

    model = SLatMeshDecoder()
    model.eval()
    print(model)