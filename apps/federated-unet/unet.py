"""Tiny U-Net for binary foreground segmentation.

GroupNorm rather than BatchNorm: FedAvg averages the whole ``state_dict``, and
averaging BatchNorm running statistics collected on sites with different image
domains degrades the merged model for reasons that have nothing to do with the
learned weights. GroupNorm carries no running state, so a merged checkpoint is
a pure function of the site weights.
"""

from collections import OrderedDict
from typing import Dict, List

import torch
from torch import nn

# Every site must build the identical architecture or the merge is meaningless.
ARCH = {"in_channels": 1, "base": 16, "depth": 4}


def _block(cin: int, cout: int, groups: int = 8) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.GroupNorm(groups, cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.GroupNorm(groups, cout),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, base: int = 16, depth: int = 4) -> None:
        super().__init__()
        channels = [base * 2**i for i in range(depth)]
        self.pool = nn.MaxPool2d(2)
        self.downs = nn.ModuleList()
        c = in_channels
        for ch in channels:
            self.downs.append(_block(c, ch))
            c = ch
        self.bottleneck = _block(c, c * 2)
        c = c * 2
        self.upconvs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for ch in reversed(channels):
            self.upconvs.append(nn.ConvTranspose2d(c, ch, 2, stride=2))
            self.ups.append(_block(ch * 2, ch))
            c = ch
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block in self.downs:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for upconv, block, skip in zip(self.upconvs, self.ups, reversed(skips)):
            x = upconv(x)
            x = block(torch.cat([skip, x], dim=1))
        return self.head(x)


def build_model(seed: int = 0) -> UNet:
    """Build the shared architecture with deterministic initialisation.

    A federated run only makes sense if round 0 starts from one common set of
    weights, so the seed has to reach the parameter initialiser itself.
    """
    torch.manual_seed(seed)
    return UNet(**ARCH)


def signature(state_dict: Dict[str, torch.Tensor]) -> List[str]:
    """Ordered ``name:shape`` list — compared before merging two checkpoints."""
    return [f"{k}:{tuple(v.shape)}" for k, v in state_dict.items()]


def fedavg(state_dicts: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
    """Sample-count-weighted average of parameter tensors (McMahan et al.)."""
    if not state_dicts:
        raise ValueError("nothing to merge")
    reference = signature(state_dicts[0])
    for i, sd in enumerate(state_dicts[1:], start=1):
        if signature(sd) != reference:
            raise ValueError(f"checkpoint {i} has a different architecture than checkpoint 0")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("merge weights must sum to a positive number")
    fractions = [w / total for w in weights]
    merged = OrderedDict()
    for key in state_dicts[0]:
        acc = state_dicts[0][key].float() * fractions[0]
        for sd, frac in zip(state_dicts[1:], fractions[1:]):
            acc = acc + sd[key].float() * frac
        merged[key] = acc
    return merged
