from typing import *
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import rembg
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
import todos
import pdb

class TrellisImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self._device = torch.device("cpu") # next(self.models['image_cond_model'].parameters()).device
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)

        # pdb.set_trace()

    def load_model(self, model_key: str) -> nn.Module:
        """Move a model to CUDA and return it."""
        # model_key -- slat_decoder_mesh ???
        # if 'slat_decoder_mesh' in model_key:
        #     pdb.set_trace()
        self.models[model_key].half().to(torch.device('cuda'))
        return self.models[model_key]

    def unload_models(self, model_keys: List[str]):
        """Unload models to CPU."""
        for key in model_keys:
            if key in self.models:
                self.models[key].to(torch.device("cpu"))
        torch.cuda.empty_cache()

    def unload_all_models(self):
        """Unload models to CPU."""
        model_keys = ['sparse_structure_decoder', 'sparse_structure_flow_model', 
            'slat_decoder_mesh', 'slat_decoder_gs', 'slat_decoder_rf', 'slat_flow_model', 'image_cond_model']
        for key in model_keys:
            if key in self.models:
                self.models[key].to(torch.device("cpu"))
        torch.cuda.empty_cache()

    @staticmethod
    def from_pretrained(path: str) -> "TrellisImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisImageTo3DPipeline, TrellisImageTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisImageTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        # args['image_cond_model'] -- 'dinov2_vitl14_reg'
        new_pipeline._init_image_cond_model(args['image_cond_model'])

        return new_pipeline
    
    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        # dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
        dinov2_model = torch.hub.load('/home/dell/.cache/torch/hub/facebookresearch_dinov2_main', 
            name, pretrained=True, trust_repo=True, source='local')
        dinov2_model.eval()

        self.models['image_cond_model'] = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            if getattr(self, 'rembg_session', None) is None:
                self.rembg_session = rembg.new_session('u2net')
            output = rembg.remove(input, session=self.rembg_session)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = output.resize((518, 518), Image.Resampling.LANCZOS)
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list): # True
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.image_cond_model_transform(image).cuda() #to(self.device)
        # features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']

        # tensor [image] size: [1, 3, 518, 518], min: -2.117904, max: 2.323529, mean: -1.657714
        features = self.load_model('image_cond_model').float()(image, is_training=True)['x_prenorm']
        # tensor [features] size: [1, 1374, 1024], min: -664.607056, max: 402.074707, mean: -0.280811
        self.unload_models(['image_cond_model'])
        patchtokens = F.layer_norm(features, features.shape[-1:]) # features.shape[-1:] == [1024]
        # tensor [patchtokens] size: [1, 1374, 1024], min: -25.644331, max: 15.487422, mean: 0.0

        return patchtokens
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # cond is dict:
        #     tensor [cond] size: [1, 1374, 1024], min: -25.644331, max: 15.487422, mean: 0.0
        #     tensor [neg_cond] size: [1, 1374, 1024], min: 0.0, max: 0.0, mean: 0.0
        # num_samples = 1
        # sampler_params = {'steps': 25, 'cfg_strength': 5.0, 'cfg_interval': [0.5, 1.0], 'rescale_t': 3.0}

        # Sample occupancy latent
        flow_model = self.load_model('sparse_structure_flow_model') # SparseStructureFlowModel,  device='cuda:0', dtype=torch.float16

        reso = flow_model.resolution # 16
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(flow_model.device)
        # tensor [noise] size: [1, 8, 16, 16, 16], min: -4.184124, max: 3.802687, mean: 0.000481
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        # sampler_params -- {'steps': 25, 'cfg_strength': 5.0, 'cfg_interval': [0.5, 1.0], 'rescale_t': 3.0}

        # self.sparse_structure_sampler -- trellis.pipelines.samplers.flow_euler.FlowEulerGuidanceIntervalSampler
        z_s = self.sparse_structure_sampler.sample(flow_model, noise, **cond, **sampler_params, verbose=True).samples
        self.unload_models(['sparse_structure_flow_model',])
        # tensor [z_s] size: [1, 8, 16, 16, 16], min: -5.693636, max: 4.097641, mean: 0.010038
        
        # Decode occupancy latent
        decoder = self.load_model('sparse_structure_decoder') # SparseStructureDecoder, 
        # tensor [decoder(z_s)] size: [1, 1, 64, 64, 64], min: -216.489441, max: 181.025513, mean: -145.066437
        # torch.argwhere(decoder(z_s)>0).size() -- [14955, 5]

        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()
        self.unload_models(['sparse_structure_decoder'])
        # tensor [coords] size: [14955, 4], min: 0.0, max: 63.0, mean: 23.262018

        return coords

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.load_model('slat_decoder_mesh')(slat) # SLatMeshDecoder
            self.unload_models(['slat_decoder_mesh'])
        if 'gaussian' in formats:
            ret['gaussian'] = self.load_model('slat_decoder_gs')(slat) # SLatGaussianDecoder
            self.unload_models(['slat_decoder_gs'])
        if 'radiance_field' in formats:
            ret['radiance_field'] = self.load_model('slat_decoder_rf')(slat)
            self.unload_models(['slat_decoder_rf'])
        torch.cuda.empty_cache()

        return ret
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        """
        # ----------------------------------------------------------------------
        # cond is dict:
        #     tensor [cond] size: [1, 1374, 1024], min: -25.644331, max: 15.487422, mean: 0.0
        #     tensor [neg_cond] size: [1, 1374, 1024], min: 0.0, max: 0.0, mean: 0.0
        # tensor [coords] size: [14955, 4], min: 0.0, max: 63.0, mean: 23.262018
        # sampler_params = {}
        # ----------------------------------------------------------------------
        flow_model = self.load_model('slat_flow_model') # SLatFlowModel
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(flow_model.device), # (14955, 8)
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        # (Pdb) sampler_params --
        # {'steps': 25, 'cfg_strength': 5.0, 'cfg_interval': [0.5, 1.0], 'rescale_t': 3.0}
        # self.slat_sampler -- 
        # <trellis.pipelines.samplers.flow_euler.FlowEulerGuidanceIntervalSampler object at 0x7f332df17c70>
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples
        self.unload_models(['slat_flow_model'])

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean

        # ----------------------------------------------------------------------
        # slat -- <trellis.modules.sparse.basic.SparseTensor object at 0x7f6a2e5c78b0>
        # tensor [slat.coords] size: [14955, 4], min: 0.0, max: 63.0, mean: 23.262018
        # tensor [slat.feats] size: [14955, 8], min: -9.589689, max: 9.938703, mean: -0.068928
        # ----------------------------------------------------------------------

        return slat

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
    ) -> dict:
        """
        Run the pipeline.
        """
        # num_samples == 1
        # sparse_structure_sampler_params == {}
        # slat_sampler_params == {}

        if preprocess_image: # True
            image = self.preprocess_image(image)
            # <PIL.Image.Image image mode=RGB size=518x518 at 0x7FBE613BE840>
        cond = self.get_cond([image])
        # cond is dict:
        #     tensor [cond] size: [1, 1374, 1024], min: -25.644333, max: 15.48742, mean: 0.0
        #     tensor [neg_cond] size: [1, 1374, 1024], min: 0.0, max: 0.0, mean: 0.0
        torch.manual_seed(seed)

        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        # tensor [coords] size: [14955, 4], min: 0.0, max: 63.0, mean: 23.262018

        slat = self.sample_slat(cond, coords, slat_sampler_params)
        # [slat] type: <class 'trellis.modules.sparse.basic.SparseTensor'>
        # tensor [slat.feats] size: [14955, 8], min: -9.590652, max: 9.931234, mean: -0.068985

        output = self.decode_slat(slat, formats)
        # output is dict:
        #     [mesh] type: <class 'list'>, [<trellis.representations.mesh.cube2mesh.MeshExtractResult object at 0x7f29a27293d0>]
        #     [gaussian] type: <class 'list'> [<trellis.representations.gaussian.gaussian_model.Gaussian object at 0x7f29a2729df0>]

        return output

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ):
        """
        Inject a sampler with multiple images as condition.
        
        Args:
            sampler_name (str): The name of the sampler to inject.
            num_images (int): The number of images to condition on.
            num_steps (int): The number of steps to run the sampler for.
        """
        sampler = getattr(self, sampler_name)
        setattr(sampler, f'_old_inference_model', sampler._inference_model)

        if mode == 'stochastic':
            if num_images > num_steps:
                print(f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m")

            cond_indices = (np.arange(num_steps) % num_images).tolist()
            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx:cond_idx+1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)
        
        elif mode =='multidiffusion':
            from .samplers import FlowEulerSampler
            def _new_inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    return pred
            
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        yield

        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, f'_old_inference_model')

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ) -> dict:
        """
        Run the pipeline with multiple images as condition

        Args:
            images (List[Image.Image]): The multi-view images of the assets
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        cond = self.get_cond(images)
        cond['neg_cond'] = cond['neg_cond'][:1]
        torch.manual_seed(seed)
        ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('sparse_structure_sampler', len(images), ss_steps, mode=mode):
            coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('slat_sampler', len(images), slat_steps, mode=mode):
            slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
