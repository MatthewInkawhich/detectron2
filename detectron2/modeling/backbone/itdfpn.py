# Copyright (c) Facebook, Inc. and its affiliates.
import math
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
from torch import nn

from detectron2.layers import Conv2d, ShapeSpec, get_norm

from .backbone import Backbone
from .build import BACKBONE_REGISTRY
from .resnet import build_resnet_backbone
from .itdresnet import build_itdresnet_backbone, resize_feature

__all__ = ["build_itdresnet_fpn_backbone", "build_retinanet_itdresnet_fpn_backbone", "ITDFPN"]


class ITDFPN(Backbone):
    def __init__(
        self, bottom_up, in_channels_list, out_channels, norm="", top_block=None, fuse_type="sum"
    ):
        super(ITDFPN, self).__init__()
        assert isinstance(bottom_up, Backbone)

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels_list, 1):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)

            lateral_conv = Conv2d(
                in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            self.add_module("fpn_lateral{}".format(idx), lateral_conv)
            self.add_module("fpn_output{}".format(idx), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.top_block = top_block
        self.bottom_up = bottom_up
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type


    def forward(self, x, config_combo):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        # First, forward thru bottom_up module
        bottom_up_features = self.bottom_up(x, config_combo)

        #print("bottom_up_features:")
        #for f in bottom_up_features:
        #    print(f.shape)

        # Reverse feature maps into top-down order (from low to high resolution)
        x = bottom_up_features[::-1]
        # Initialize results
        results = []
        # Forward thru top of pyramid (not counting p6, p7), record in results
        prev_features = self.lateral_convs[0](x[0])
        results.append(self.output_convs[0](prev_features))
        # Iterate over rest of features
        for features, lateral_conv, output_conv in zip(
            x[1:], self.lateral_convs[1:], self.output_convs[1:]
        ):
            # Resize prev_features to match current features 
            top_down_features = resize_feature(prev_features, features.shape[-2:])
            # Forward current features thru current lateral_conv
            lateral_features = lateral_conv(features)
            # Fuse lateral features with top down features
            prev_features = lateral_features + top_down_features
            if self._fuse_type == "avg":
                prev_features /= 2
            # Insert in front of results
            results.insert(0, output_conv(prev_features))
        # Forward thru top block and record
        if self.top_block is not None:
            if isinstance(self.top_block, LastLevelP6P7):
                results.extend(self.top_block(x[0]))
            elif isinstance(self.top_block, LastLevelMaxPool):
                results.extend(self.top_block(results[-1]))
        return results


class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """
    def __init__(self):
        super().__init__()
        self.num_levels = 1

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from
    C5 feature.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.num_levels = 2
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            weight_init.c2_xavier_fill(module)

    def forward(self, c5):
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


@BACKBONE_REGISTRY.register()
def build_itdresnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    # Build ITD ResNet
    bottom_up = build_itdresnet_backbone(cfg, input_shape)
    # Create in_channels_list
    body_channels = cfg.MODEL.ITD_BACKBONE.BODY_CHANNELS
    return_features = cfg.MODEL.ITD_BACKBONE.RETURN_FEATURES
    in_channels_list = [body_channels[i][2] for i in range(len(body_channels)) if return_features[i]]
    # Initialize other arg variables
    out_channels = cfg.MODEL.ITD_FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        norm=cfg.MODEL.ITD_FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.ITD_FPN.FUSE_TYPE,
    )
    return backbone


@BACKBONE_REGISTRY.register()
def build_retinanet_itdresnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    # Build ITD ResNet
    bottom_up = build_itdresnet_backbone(cfg, input_shape)
    # Create in_channels_list
    body_channels = cfg.MODEL.ITD_BACKBONE.BODY_CHANNELS
    return_features = cfg.MODEL.ITD_BACKBONE.RETURN_FEATURES
    in_channels_list = [body_channels[i][2] for i in range(len(body_channels)) if return_features[i]]
    # Initialize other arg variables
    out_channels = cfg.MODEL.ITD_FPN.OUT_CHANNELS
    in_channels_p6p7 = in_channels_list[-1]
    #print("in_channels_list:", in_channels_list)
    #print("out_channels:", out_channels)
    #print("in_channels_p6p7:", in_channels_p6p7)
    #exit()

    # Build ITDFPN module
    backbone = ITDFPN(
        bottom_up=bottom_up,
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        norm=cfg.MODEL.ITD_FPN.NORM,
        top_block=LastLevelP6P7(in_channels_p6p7, out_channels),
        fuse_type=cfg.MODEL.ITD_FPN.FUSE_TYPE,
    )
    return backbone
