# https://zhuanlan.zhihu.com/p/30989191234

import pdb
import os
os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("./JeffreyXiang/TRELLIS-image-large")
#pipeline.cuda()
pdb.set_trace()

# for k, v in pipeline.models.items(): print(k, " -- ", v.__class__)
# --------------------------------------------------------------------------------------------------------------
# 0) image_cond_model  --  <class 'dinov2.models.vision_transformer.DinoVisionTransformer'>
# 1) sparse_structure_flow_model  --  <class 'trellis.models.sparse_structure_flow.SparseStructureFlowModel'>
#    sparse_structure_decoder  --  <class 'trellis.models.sparse_structure_vae.SparseStructureDecoder'>
# 2) slat_flow_model  --  <class 'trellis.models.structured_latent_flow.SLatFlowModel'>
# 3) slat_decoder_mesh  --  <class 'trellis.models.structured_latent_vae.decoder_mesh.SLatMeshDecoder'>
# 4) slat_decoder_gs  --  <class 'trellis.models.structured_latent_vae.decoder_gs.SLatGaussianDecoder'>

# slat_decoder_rf  --  <class 'trellis.models.structured_latent_vae.decoder_rf.SLatRadianceFieldDecoder'>
# --------------------------------------------------------------------------------------------------------------

# Load an image
image = Image.open("assets/example_image/T.png")
# image = Image.open("assets/example_image/typical_vehicle_bulldozer.png")

# Run the pipeline
outputs = pipeline.run(
    image,
    seed=1,
    formats = ['mesh', 'gaussian'],

    # Optional parameters
    # sparse_structure_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 7.5,
    # },
    # slat_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 3,
    # },
)
# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['gaussian']: a list of 3D Gaussians
# - outputs['radiance_field']: a list of radiance fields
# - outputs['mesh']: a list of meshes

# Render the outputs
# video = render_utils.render_video(outputs['gaussian'][0])['color']
# imageio.mimsave("sample_gs.mp4", video, fps=30)
# video = render_utils.render_video(outputs['radiance_field'][0])['color']
# imageio.mimsave("sample_rf.mp4", video, fps=30)
# video = render_utils.render_video(outputs['mesh'][0])['normal']
# imageio.mimsave("sample_mesh.mp4", video, fps=30)

# GLB files can be extracted from the outputs
glb = postprocessing_utils.to_glb(
    outputs['gaussian'][0],
    outputs['mesh'][0],
    # Optional parameters
    simplify=0.95,          # Ratio of triangles to remove in the simplification process
    texture_size=1024,      # Size of the texture used for the GLB
)
glb.export("sample.glb")

# # Save Gaussians as PLY files
# outputs['gaussian'][0].save_ply("sample.ply")
