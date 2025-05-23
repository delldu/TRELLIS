from typing import *
import pdb

class GuidanceIntervalSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance with interval.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
        # cfg_interval -- [0.5, 1.0]

        if cfg_interval[0] <= t <= cfg_interval[1]:
            # pdb.set_trace()
            # t -- 1.0
            # cfg_strength -- 5.0
            pred = super()._inference_model(model, x_t, t, cond, **kwargs)
            neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred
        else:
            # ==> pdb.set_trace()
            return super()._inference_model(model, x_t, t, cond, **kwargs)
