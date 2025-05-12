import torch
from ...modules.sparse import SparseTensor
from easydict import EasyDict as edict
from .utils_cube import *
from .flexicubes.flexicubes import FlexiCubes
import pdb

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
                
    # def comput_v_normals(self, verts, faces):
    #     i0 = faces[..., 0].long()
    #     i1 = faces[..., 1].long()
    #     i2 = faces[..., 2].long()

    #     v0 = verts[i0, :]
    #     v1 = verts[i1, :]
    #     v2 = verts[i2, :]
    #     face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    #     v_normals = torch.zeros_like(verts)
    #     v_normals.scatter_add_(0, i0[..., None].repeat(1, 3), face_normals)
    #     v_normals.scatter_add_(0, i1[..., None].repeat(1, 3), face_normals)
    #     v_normals.scatter_add_(0, i2[..., None].repeat(1, 3), face_normals)

    #     v_normals = torch.nn.functional.normalize(v_normals, dim=1)
    #     return v_normals   

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
        ## tensor [verts] size: [16974593, 3], min: 0.0, max: 256.0, mean: 128.0
        ## tensor [cube] size: [16777216, 8], min: 0.0, max: 16974592.0, mean: 8487296.0
        self.reg_c = cube.to(self.device) # size: [16777216, 8]
        self.reg_v = verts.to(self.device) # size: [16974593, 3]
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
        # self.layouts -------------------------------------------------
        # sdf {'shape': [8, 1], 'size': 8, 'range': [0, 8]}
        # deform {'shape': [8, 3], 'size': 24, 'range': [8, 32]}
        # weights {'shape': [21], 'size': 21, 'range': [32, 53]}
        # color {'shape': [8, 6], 'size': 48, 'range': [53, 101]}
        # self.layouts -------------------------------------------------
        start = 0
        for k, v in self.layouts.items():
            v['range'] = (start, start + v['size'])
            start += v['size']
        self.feats_channels = start # 101 

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
        coords = cubefeats.coords[:, 1:] # size(): [957120, 4] --> [957120, 3]
        feats = cubefeats.feats # size() -- ([957120, 101] ??? 101
        
        sdf, deform, color, weights = [self.get_layout(feats, name) for name in ['sdf', 'deform', 'color', 'weights']]
        sdf += self.sdf_bias

        assert  self.use_color == True
        v_attrs = [sdf, deform, color] if self.use_color else [sdf, deform]
        # v_attrs is list: len = 3
        #     tensor [sdf] size: [957120, 8, 1], min: -0.144015, max: 0.257282, mean: 0.003073
        #     tensor [deform] size: [957120, 8, 3], min: -7.113086, max: 6.312462, mean: 0.00125
        #     tensor [color] size: [957120, 8, 6], min: -13.710188, max: 9.915195, mean: -0.954984

        v_pos, v_attrs, reg_loss = sparse_cube2verts(coords, torch.cat(v_attrs, dim=-1), training=training)
        v_attrs_d = get_dense_attrs(v_pos, v_attrs, res=self.res+1, sdf_init=True)

        # todos.debug.output_var("weights", weights)
        # tensor [weights] size: [957120, 21], min: -11.96298, max: 16.188641, mean: 0.010774
        weights_d = get_dense_attrs(coords, weights, res=self.res, sdf_init=False)
        # tensor [weights_d] size: [16777216, 21], min: -11.96298, max: 16.188641, mean: 0.000615

        if self.use_color: # True
            sdf_d, deform_d, colors_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4], v_attrs_d[..., 4:]
        else:
            sdf_d, deform_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4]
            colors_d = None

        x_nx3 = get_defomed_verts(self.reg_v, deform_d, self.res)
        # tensor [x_nx3] size: [16974593, 3], min: -0.500935, max: 0.50072, mean: -2e-06

        # tensor [sdf_d] size: [16974593], min: -0.138959, max: 1.0, mean: 0.934898
        vertices, faces, L_dev, colors = self.mesh_extractor(
            voxelgrid_vertices=x_nx3,
            scalar_field=sdf_d,
            cube_idx=self.reg_c,
            resolution=self.res, # 256
            beta=weights_d[:, :12],
            alpha=weights_d[:, 12:20],
            gamma_f=weights_d[:, 20],
            voxelgrid_colors=colors_d,
            training=training) # training === False
        
        # ===> pdb.set_trace()
        mesh = MeshExtractResult(vertices=vertices, faces=faces, vertex_attrs=colors, res=self.res)
        if training: # False
            if mesh.success:
                reg_loss += L_dev.mean() * 0.5
            reg_loss += (weights[:,:20]).abs().mean() * 0.2
            mesh.reg_loss = reg_loss
            mesh.tsdf_v = get_defomed_verts(v_pos, v_attrs[:, 1:4], self.res)
            mesh.tsdf_s = v_attrs[:, 0]
        return mesh
