"""Prompt-free bioimage.io architecture for micro-SAM Automatic Instance Segmentation.

A single ``nn.Module`` that maps an RGB image to the three AIS decoder maps
(foreground, center distance, boundary distance) with the SAM preprocessing baked
in, so ``bioimageio.core`` can run a plain image-in -> maps-out prediction pass.
The watershed that turns the three maps into instance labels is downstream and is
not part of this module (it is not expressible as a spec postprocessing op). This
file ships inside the exported package as the ``pytorch_state_dict`` architecture
source, so it must stay import-light and self-contained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _preprocess_shape(oldh: int, oldw: int, target_length: int):
    scale = target_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    return int(newh + 0.5), int(neww + 0.5)


class MicroSAMAIS(nn.Module):
    def __init__(self, model_type: str = "vit_b", use_conv_transpose: bool = False):
        super().__init__()
        from micro_sam.util import get_sam_model
        from micro_sam.instance_segmentation import DecoderAdapter
        from torch_em.model import UNETR

        predictor = get_sam_model(model_type=model_type, device="cpu")
        image_encoder = predictor.model.image_encoder
        unetr = UNETR(
            backbone="sam", encoder=image_encoder, out_channels=3,
            use_sam_stats=True, final_activation="Sigmoid",
            use_skip_connection=False, resize_input=True,
            use_conv_transpose=use_conv_transpose,
        )
        self.image_encoder = image_encoder
        self.decoder = DecoderAdapter(unetr)
        self.register_buffer("pixel_mean", predictor.model.pixel_mean.clone())
        self.register_buffer("pixel_std", predictor.model.pixel_std.clone())
        self.img_size = int(image_encoder.img_size)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        _, _, h, w = image.shape
        nh, nw = _preprocess_shape(h, w, self.img_size)
        x = F.interpolate(image, size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)
        x = (x - self.pixel_mean) / self.pixel_std
        x = F.pad(x, (0, self.img_size - nw, 0, self.img_size - nh))
        embeddings = self.image_encoder(x)
        maps = self.decoder(embeddings, (nh, nw), (h, w))
        return maps
