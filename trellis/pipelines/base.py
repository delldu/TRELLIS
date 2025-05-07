from typing import *
import torch
import torch.nn as nn
from .. import models
import pdb

class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()

    @staticmethod
    def from_pretrained(path: str) -> "Pipeline":
        """
        Load a pretrained model.
        """
        import os
        import json
        is_local = os.path.exists(f"{path}/pipeline.json")

        if is_local:
            config_file = f"{path}/pipeline.json"
        else:
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(path, "pipeline.json")

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        # sparse_structure_decoder ===> ckpts/ss_dec_conv3d_16l8_fp16
        # sparse_structure_flow_model ===> ckpts/ss_flow_img_dit_L_16l8_fp16
        # slat_decoder_gs ===> ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16
        # slat_decoder_rf ===> ckpts/slat_dec_rf_swin8_B_64l8r16_fp16
        # slat_decoder_mesh ===> ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16, SLatMeshDecoder
        # slat_flow_model ===> ckpts/slat_flow_img_dit_L_64l8p2_fp16

        _models = {}
        for k, v in args['models'].items():
            print(k, "====>", f"{path}/{v} ...")
            try:
                _models[k] = models.from_pretrained(f"{path}/{v}")
                print(k, "====>", f"{path}/{v} ... OK")
            except:
                print(k, "====>", f"{path}/{v} ... NOK")
                # _models[k] = models.from_pretrained(v)
                # pdb.set_trace()

            # print("-" * 80)

        new_pipeline = Pipeline(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    @property
    def device(self) -> torch.device:
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))
