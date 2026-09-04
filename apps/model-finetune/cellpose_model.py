"""BioImage.IO wrapper for a fine-tuned Cellpose model (cpsam or cpdino).

Bundled verbatim into every Cellpose export as ``model.py`` — the exported RDF's
``weights.pytorch_state_dict.architecture`` points ``callable: CellposeSAMWrapper``
here, and bioimageio.core imports this file to instantiate the model. It wraps
the raw Cellpose Transformer (``net``) as an ``nn.Module`` whose ``forward`` runs
cellpose's flow-dynamics postprocessing and returns an instance-label mask, so the
package self-tests and serves without a separate postprocessing op. ``model_type``
selects the backbone: ``cpsam`` → CPSAM (SAM ViT-L), ``cpdino`` → CPDINO ViT-L,
``cpdino-vitb`` → CPDINO ViT-B.

The exported ``pytorch_state_dict`` is the bare ``net`` state dict (what
``net.save_model`` writes during training), loaded via ``load_state_dict`` below.
Kept in sync with ``apps/cellpose-finetuning/model_template.py``.
"""
import numpy as np
import torch
import torch.nn as nn
from cellpose import models as cpmodels
from cellpose.core import assign_device


# nn.Module.eval() and CellposeModel.eval() collide under multiple inheritance;
# preserve the cellpose predictor under a different name before wrapping.
cpmodels.CellposeModel.evaluate = cpmodels.CellposeModel.eval  # type: ignore


class CellposeSAMWrapper(nn.Module, cpmodels.CellposeModel):
    """Cellpose-SAM as a BioImage.IO-compatible ``nn.Module``."""

    def __init__(
        self,
        model_type="cpsam",
        diam_mean=30.0,
        cp_batch_size=8,
        channels=[0, 0],
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        stitch_threshold=0.0,
        estimate_diam=False,
        normalize=True,
        do_3D=False,
        gpu=True,
        use_bfloat16=True,
    ):
        nn.Module.__init__(self)

        self.model_type = model_type
        self.diam_mean = diam_mean
        self.cp_batch_size = cp_batch_size
        self.channels = channels
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.stitch_threshold = stitch_threshold
        self.estimate_diam = estimate_diam
        self.normalize = normalize
        self.do_3D = do_3D
        self.use_bfloat16 = use_bfloat16

        self.device, self.gpu = assign_device(use_torch=True, gpu=gpu)

        dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        # CellposeModel.eval / _run_net read self.backbone (cellpose 4.2.x) to pick
        # the tile size (256 for cpsam, 384 for cpdino). We bypass
        # CellposeModel.__init__, so set net + backbone by hand.
        if model_type in ("cpdino", "cpdino-vitb"):
            from cellpose.vit import CPDINO

            model_name = "vitb" if model_type == "cpdino-vitb" else "vitl"
            self.net = CPDINO(model_name=model_name, dtype=dtype).to(self.device)
            self.backbone = "dino_" + model_name
        else:
            from cellpose.vit import CPSAM

            self.net = CPSAM(dtype=dtype).to(self.device)
            self.backbone = "sam_vitl"

        self.net.diam_labels = nn.Parameter(torch.tensor([diam_mean]), requires_grad=False)
        self.net.diam_mean = nn.Parameter(torch.tensor([diam_mean]), requires_grad=False)

        self.nclasses = 3
        self.channel_axis = None

    def load_state_dict(self, state_dict, strict=True, assign=False):
        from collections import namedtuple

        Incompatible = namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])

        result = self.net.load_state_dict(state_dict, strict=strict)

        if hasattr(self.net, "diam_mean"):
            self.diam_mean = self.net.diam_mean.data.cpu().numpy()[0]
        if hasattr(self.net, "diam_labels"):
            self.diam_labels = self.net.diam_labels.data.cpu().numpy()[0]

        return result

    def eval(self, *args, **kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            return self.train(False)
        return self.evaluate(*args, **kwargs)  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 4:
            raise ValueError(
                f"Input image(s) must be 4-dimensional (batch, channel, height, width), "
                f"got shape {x.shape}"
            )

        image_list = []
        for img in x:
            img_np = img.permute(1, 2, 0).cpu().float().numpy()
            if img_np.shape[2] == 1:
                img_np = np.concatenate([img_np, img_np, img_np], axis=2)
            elif img_np.shape[2] == 2:
                img_np = np.concatenate([img_np, np.zeros_like(img_np[:, :, 0:1])], axis=2)
            elif img_np.shape[2] > 3:
                img_np = img_np[:, :, :3]
            image_list.append(img_np)

        masks_list, flows_list, styles_list = self.eval(  # type: ignore
            image_list,
            channel_axis=self.channel_axis,
            diameter=self.diam_mean,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            stitch_threshold=self.stitch_threshold,
            batch_size=self.cp_batch_size,
            normalize=self.normalize,
            do_3D=self.do_3D,
        )

        if isinstance(masks_list, list):
            masks = torch.stack([torch.from_numpy(np.array(m, dtype=np.float32)) for m in masks_list])
        else:
            masks = torch.from_numpy(np.array(masks_list, dtype=np.float32))

        masks = masks.to(x.device)
        if len(masks.shape) == 2:
            masks = masks.unsqueeze(0)
        return masks
