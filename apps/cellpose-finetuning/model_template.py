"""BioImage.io Model Wrapper for Cellpose 4.0.7 (Cellpose-SAM).

This wrapper provides a PyTorch nn.Module interface for Cellpose 4.0.7 models
that is compatible with the BioImage.io model format.
"""

import numpy as np
import torch
import torch.nn as nn
from cellpose import models as cpmodels
from cellpose.vit_sam import Transformer
from cellpose.core import assign_device


# Prevent mix-up between pytorch module eval and cellpose eval functions
cpmodels.CellposeModel.evaluate = cpmodels.CellposeModel.eval  # type: ignore


class CellposeSAMWrapper(nn.Module, cpmodels.CellposeModel):
    """
    A wrapper around the Cellpose 4.0.7 (Cellpose-SAM) model
    which acts as a PyTorch model compatible with BioImage.io format.

    This wrapper is designed for the Transformer-based Cellpose-SAM architecture.
    """

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
        """Initialize the Cellpose-SAM wrapper.

        Args:
            model_type: Model type (default: "cpsam" for Cellpose-SAM)
            diam_mean: Mean diameter of objects (default: 30.0 pixels)
            cp_batch_size: Batch size for cellpose processing (default: 8)
            channels: Channel configuration [cytoplasm, nucleus] (default: [0, 0] for grayscale)
            flow_threshold: Flow error threshold for mask reconstruction (default: 0.4)
            cellprob_threshold: Cell probability threshold (default: 0.0)
            stitch_threshold: Threshold for stitching tiles (default: 0.0)
            estimate_diam: Whether to estimate diameter automatically (default: False)
            normalize: Whether to normalize images (default: True)
            do_3D: Whether to process 3D images (default: False)
            gpu: Whether to use GPU (default: True)
            use_bfloat16: Whether to use bfloat16 precision (default: True)
        """
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

        # Device assignment
        self.device, self.gpu = assign_device(use_torch=True, gpu=gpu)

        # Create Transformer network (Cellpose-SAM)
        dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        self.net = Transformer(dtype=dtype).to(self.device)

        # Set diameter parameters
        self.net.diam_labels = nn.Parameter(
            torch.tensor([diam_mean]), requires_grad=False
        )
        self.net.diam_mean = nn.Parameter(
            torch.tensor([diam_mean]), requires_grad=False
        )

        # Cellpose model parameters
        self.nclasses = 3
        self.channel_axis = None
        self.invert = False

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load model weights from state dict.

        Args:
            state_dict: Dictionary containing model weights
            strict: Whether to strictly enforce key matching (default: True)
            assign: Whether to assign values (default: False)

        Returns:
            NamedTuple with missing_keys and unexpected_keys
        """
        from collections import namedtuple

        Incompatible = namedtuple(
            "IncompatibleKeys", ["missing_keys", "unexpected_keys"]
        )

        # Load the state dict into the network
        result = self.net.load_state_dict(state_dict, strict=strict)

        # Update diameter parameters from loaded weights
        if hasattr(self.net, "diam_mean"):
            self.diam_mean = self.net.diam_mean.data.cpu().numpy()[0]
        if hasattr(self.net, "diam_labels"):
            self.diam_labels = self.net.diam_labels.data.cpu().numpy()[0]

        return result

    def eval(self, *args, **kwargs):
        """Evaluate the model.

        This method handles both PyTorch module eval (no args) and
        Cellpose eval (with args) by dispatching appropriately.
        """
        if len(args) == 0 and len(kwargs) == 0:
            # PyTorch module eval
            return self.train(False)
        else:
            # Cellpose model eval
            return self.evaluate(*args, **kwargs)  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for BioImage.io compatibility.

        Args:
            x: Input tensor of shape (batch, channel, height, width)

        Returns:
            masks: Segmentation masks of shape (batch, height, width)

        Raises:
            ValueError: If input dimensions are invalid
        """
        if len(x.shape) != 4:
            raise ValueError(
                f"Input image(s) must be 4-dimensional (batch, channel, height, width), "
                f"got shape {x.shape}"
            )

        # Convert torch tensor to list of numpy arrays (Y, X, C format for cellpose)
        image_list = []
        for img in x:
            # Convert from (C, H, W) to (H, W, C)
            img_np = img.permute(1, 2, 0).cpu().numpy()

            # Ensure 3 channels for Cellpose-SAM
            if img_np.shape[2] == 1:
                # Replicate single channel to 3 channels
                img_np = np.concatenate([img_np, img_np, img_np], axis=2)
            elif img_np.shape[2] == 2:
                # Add a zero channel
                img_np = np.concatenate(
                    [img_np, np.zeros_like(img_np[:, :, 0:1])], axis=2
                )
            elif img_np.shape[2] > 3:
                # Use first 3 channels
                img_np = img_np[:, :, :3]

            image_list.append(img_np)

        # Run cellpose eval
        masks_list, flows_list, styles_list = self.eval(  # type: ignore
            image_list,
            channels=self.channels,
            channel_axis=self.channel_axis,
            diameter=self.diam_mean,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            stitch_threshold=self.stitch_threshold,
            batch_size=self.cp_batch_size,
            normalize=self.normalize,
            invert=self.invert,
            do_3D=self.do_3D,
        )

        # Convert masks to tensor
        if isinstance(masks_list, list):
            masks = torch.stack(
                [torch.from_numpy(np.array(m, dtype=np.float32)) for m in masks_list]
            )
        else:
            masks = torch.from_numpy(np.array(masks_list, dtype=np.float32))

        # Move to same device as input
        masks = masks.to(x.device)

        # Ensure correct shape (B, H, W)
        if len(masks.shape) == 2:
            masks = masks.unsqueeze(0)

        return masks


if __name__ == "__main__":
    """Test the wrapper."""
    import matplotlib.pyplot as plt

    # Create a test image
    test_image = torch.randn(1, 3, 256, 256)

    # Initialize model
    model = CellposeSAMWrapper(diam_mean=30.0, gpu=False, use_bfloat16=False)

    # Note: To actually use this, you need to load trained weights:
    # model.load_state_dict(torch.load("path/to/model"))

    print(f"Model initialized successfully")
    print(f"Input shape: {test_image.shape}")

    # Forward pass (requires loaded weights to produce meaningful output)
    # masks = model(test_image)
    # print(f"Output shape: {masks.shape}")
