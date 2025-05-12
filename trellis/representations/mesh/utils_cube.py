import torch
import todos
import pdb

cube_corners = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [
        1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.int)
# cube_neighbor = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
# cube_edges = torch.tensor([0, 1, 1, 5, 4, 5, 0, 4, 2, 3, 3, 7, 6, 7, 2, 6,
#                 2, 0, 3, 1, 7, 5, 6, 4], dtype=torch.long, requires_grad=False)

# --------------------------------------------------------------        
def construct_dense_grid(res, device='cuda'):
    '''construct a dense grid based on resolution'''
    res_v = res + 1
    assert res_v == 257

    vertsid = torch.arange(res_v ** 3, device=device)
    # tensor [vertsid] size: [16974593], min: 0.0, max: 16974592.0, mean: 8487297.0

    coordsid = vertsid.reshape(res_v, res_v, res_v)[:res, :res, :res].flatten()
    # tensor [coordsid] size: [16777216], min: 0.0, max: 16908284.0, mean: 8454142.0

    # (Pdb) cube_corners.size() -- [8, 3]
    # (Pdb) cube_corners --
    # tensor([[0, 0, 0],
    #         [1, 0, 0],
    #         [0, 1, 0],
    #         [1, 1, 0],
    #         [0, 0, 1],
    #         [1, 0, 1],
    #         [0, 1, 1],
    #         [1, 1, 1]], dtype=torch.int32)
    cube_corners_bias = (cube_corners[:, 0] * res_v + cube_corners[:, 1]) * res_v + cube_corners[:, 2]
    # tensor [cube_corners_bias] size: [8], min: 0.0, max: 66307.0, mean: 33153.5

    cube_fx8 = (coordsid.unsqueeze(1) + cube_corners_bias.unsqueeze(0).to(device))
    ## tensor [cube_fx8] size: [16777216, 8], min: 0.0, max: 16974592.0, mean: 8487296.0

    verts = torch.stack([vertsid // (res_v ** 2), (vertsid // res_v) % res_v, vertsid % res_v], dim=1)
    ## tensor [verts] size: [16974593, 3], min: 0.0, max: 256.0, mean: 128.0

    return verts, cube_fx8

# --------------------------------------------------------------        
def construct_voxel_grid(coords):
    # tensor [coords] size: [957120, 3], min: 0.0, max: 255.0, mean: 125.56411
    verts = (cube_corners.unsqueeze(0).to(coords) + coords.unsqueeze(1)).reshape(-1, 3)
    # tensor [verts] size: [7656960, 3], min: 0.0, max: 256.0, mean: 126.06411

    verts_unique, inverse_indices = torch.unique(verts, dim=0, return_inverse=True)
    # tensor [verts_unique] size: [1112881, 3], min: 0.0, max: 256.0, mean: 126.11235
    # tensor [inverse_indices] size: [7656960], min: 0.0, max: 1112880.0, mean: 557108.375

    cubes = inverse_indices.reshape(-1, 8)
    # tensor [cubes] size: [957120, 8], min: 0.0, max: 1112880.0, mean: 557108.375

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
    # num_verts -- 1112881
    # tensor [cubes] size: [957120, 8], min: 0.0, max: 1112880.0, mean: 557108.375
    # tensor [value] size: [957120, 8, 10], min: -13.710188, max: 9.915195, mean: -0.572308

    M = value.shape[2] # number of channels === 10
    assert M == 10
    reduced = torch.zeros(num_verts, M, device=cubes.device)
    # tensor [reduced] size: [1112881, 10], min: 0.0, max: 0.0, mean: 0.0

    # todos.debug.output_var("cubes", cubes.unsqueeze(-1).expand(-1, -1, M).flatten(0, 1))
    # todos.debug.output_var("value", value.flatten(0, 1))
    # tensor [cubes] size: [7656960, 10], min: 0.0, max: 1112880.0, mean: 557108.375
    # tensor [value] size: [7656960, 10], min: -13.710188, max: 9.915195, mean: -0.572308

    return torch.scatter_reduce(reduced, 0, 
        cubes.unsqueeze(-1).expand(-1, -1, M).flatten(0, 1), 
        value.flatten(0, 1), reduce=reduce, include_self=False)

# --------------------------------------------------------------        
def sparse_cube2verts(coords, feats, training=True):
    new_coords, cubes = construct_voxel_grid(coords)
    new_feats = cubes_to_verts(new_coords.shape[0], cubes, feats)
    # tensor [new_coords] size: [1112881, 3], min: 0.0, max: 256.0, mean: 126.11235
    # tensor [new_feats] size: [1112881, 10], min: -13.703279, max: 9.91334, mean: -0.56025

    # assert training == True or ...
    if training:
        con_loss = torch.mean((feats - new_feats[cubes]) ** 2)
    else:
        con_loss = 0.0
    return new_coords, new_feats, con_loss
    
# --------------------------------------------------------------        
def get_dense_attrs(coords : torch.Tensor, feats : torch.Tensor, res : int, sdf_init=True):
    # --------------------------------------------------------------        
    # tensor [coords] size: [1112881, 3], min: 0.0, max: 256.0, mean: 126.11235
    # tensor [feats] size: [1112881, 10], min: -13.709862, max: 9.92256, mean: -0.560406
    # res = 257
    # sdf_init = True
    # --------------------------------------------------------------        

    F = feats.shape[-1]
    dense_attrs = torch.zeros([res] * 3 + [F], device=feats.device)
    # tensor [dense_attrs] size: [257, 257, 257, 10], min: 0.0, max: 0.0, mean: 0.0

    # assert sdf_init == False or ...
    if sdf_init:
        dense_attrs[..., 0] = 1 # initial outside sdf value

    # tensor [feats] size: [1112881, 10], min: -13.709862, max: 9.92256, mean: -0.560406
    dense_attrs[coords[:, 0], coords[:, 1], coords[:, 2], :] = feats
    # tensor [dense_attrs] size: [257, 257, 257, 10], min: -13.709862, max: 9.92256, mean: 0.056703

    return dense_attrs.reshape(-1, F) # size() -- [16974593, 10]

# --------------------------------------------------------------        
def get_defomed_verts(v_pos : torch.Tensor, deform : torch.Tensor, res):
    # tensor [v_pos] size: [16974593, 3], min: 0.0, max: 256.0, mean: 128.0
    # tensor [deform] size: [16974593, 3], min: -7.126225, max: 6.292705, mean: 0.000398
    # res = 256

    return v_pos / res - 0.5 + (1 - 1e-8) / (res * 2) * torch.tanh(deform)
        