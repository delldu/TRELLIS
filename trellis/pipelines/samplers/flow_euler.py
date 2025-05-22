from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin
import pdb

class FlowEulerSampler(Sampler):
    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        # tensor [x_t] size: [1, 8, 16, 16, 16], min: -4.184124, max: 3.802687, mean: 0.000481
        # t --- 1.0, ... 0.0
        assert kwargs == {}

        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=x_t.dtype)
        # tensor [cond] size: [1, 1374, 1024], min: -25.644331, max: 15.487422, mean: 0.0
        if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1: # False
            cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))

        # model -- SparseStructureFlowModel, device='cuda:0', dtype=torch.float16
        return model(x_t, t, cond, **kwargs)


    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return pred_x_prev

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)

        assert rescale_t == 3.0
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)

        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None})

        # model -- SparseStructureFlowModel, cuda, torch.float16
        # model = model.float()

        for t, t_prev in tqdm(t_pairs, desc="FlowEulerSampler Sampling", disable=not verbose):
            sample = self.sample_once(model, sample, t, t_prev, cond, **kwargs)

        ret.samples = sample
        return ret


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, **kwargs)


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        return super().sample(model, noise, cond, steps, rescale_t, verbose, 
                neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)
