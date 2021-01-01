# MatthewInkawhich

import logging
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import (
    CNNBlockBase,
    Conv2d,
    DeformConv,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)

from .backbone import Backbone
from .build import BACKBONE_REGISTRY

__all__ = [
    "ITDBottleneckBlock",
    "Stem",
    "ITDResNet",
    "build_itdresnet_backbone",
]



"""
Takes a torch tensor (feature) of shape NxCxHxW, and resizes the resolution
until it matches target_resolution. Note that target_resolution height/width 
should be larger/smaller by a factor of 2.
"""
def resize_feature(x, target_resolution):
    # If x's shape already matches target, return
    if x.shape[-2:][0] == target_resolution[0] and x.shape[-2:][1] == target_resolution[1]:
        return x
    # If x's H or W is smaller than target's, interpolate up
    elif x.shape[-2:][0] < target_resolution[0] or x.shape[-2:][1] < target_resolution[1]:
        return F.interpolate(x, size=target_resolution, mode='nearest')
    # x's H/W are larger than target, pool to size
    else:
        return F.adaptive_avg_pool2d(x, target_resolution)



class ITDBottleneckBlock(CNNBlockBase):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        stride_options,
        dilation_options,
        ksize_options,
        norm="BN",
    ):
        # Stride is dynamic at run time, so just use 1
        super().__init__(in_channels, out_channels, 1)

        ### Residual layer
        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None
           
        ### Conv1
        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )
        
        ### Conv2 
        self.conv2_stride_options = stride_options
        self.conv2_dilation_options = dilation_options
        self.conv2_ksize_options = ksize_options
        self.conv2_weight = nn.Parameter(
            torch.Tensor(bottleneck_channels, bottleneck_channels, 3, 3))
        self.bn2 = get_norm(norm, bottleneck_channels)

        ### Conv3
        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        # Weight initialization
        for layer in [self.conv1, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)
        nn.init.kaiming_normal_(self.conv2_weight, mode='fan_out', nonlinearity='relu')
    

    def forward(self, x, stride_choice, dilation_choice, ksize_choice):
        # Store input to variable
        shortcut = x

        # Forward thru conv1
        out = self.conv1(x)
        conv1_out = F.relu(out)
    
        # Prepare conv2
        use_tconv = self.conv2_stride_options[stride_choice][0]
        stride = tuple(self.conv2_stride_options[stride_choice][1])
        dilation = tuple(self.conv2_dilation_options[dilation_choice])
        ksize = tuple(self.conv2_ksize_options[ksize_choice])
        # Interpolate kernel size
        resized_conv2_weight = F.interpolate(self.conv2_weight, size=ksize, mode='bilinear', align_corners=False)
        padding = ((ksize[0] // 2) * dilation[0], (ksize[1] // 2) * dilation[1])

        # Forward thru conv2
        if use_tconv:
            output_padding = (stride[0]-1, stride[1]-1)
            out = F.conv_transpose2d(conv1_out, resized_conv2_weight.flip([2, 3]).permute(1, 0, 2, 3), stride=stride, padding=padding, output_padding=output_padding, dilation=dilation)
        else:
            out = F.conv2d(conv1_out, resized_conv2_weight, stride=stride, padding=padding, dilation=dilation)
        # Scale activations to account for kernel size interpolation
        out = out * ((3 ** 2) / (ksize[0] * ksize[1]))
        out = self.bn2(out)
        out = F.relu_(out)

        # Conv3 stage
        out = self.conv3(out)

        # Add residual
        if self.shortcut is not None:
            shortcut = self.shortcut(shortcut)
        shortcut = resize_feature(shortcut, out.shape[-2:])
        out += shortcut
        out = F.relu(out)

        return out



class Stem(CNNBlockBase):
    def __init__(self, in_channels=3, out_channels=64, norm="BN"):
        super().__init__(in_channels, out_channels, 4)
        self.in_channels = in_channels
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            norm=get_norm(norm, out_channels),
        )
        weight_init.c2_msra_fill(self.conv1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x



class ITDResNet(Backbone):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            cfg object which contains necessary configs for model building and running
        """
        super().__init__()

        self.norm_func = cfg.MODEL.ITD_BACKBONE.NORM
        # Construct Stem
        self.stem = Stem(in_channels, cfg.MODEL.ITD_BACKBONE.STEM_CHANNELS, self.norm_func)

        # Construct Blocks
        self.block_names = []
        self.return_feature_map = {}
        body_channels = cfg.MODEL.ITD_BACKBONE.BODY_CHANNELS
        stride_options = cfg.MODEL.ITD_BACKBONE.STRIDE_OPTIONS
        dilation_options = cfg.MODEL.ITD_BACKBONE.DILATION_OPTIONS
        ksize_options = cfg.MODEL.ITD_BACKBONE.KSIZE_OPTIONS
        return_features = cfg.MODEL.ITD_BACKBONE.RETURN_FEATURES
        self.stride_config = cfg.MODEL.ITD_BACKBONE.STRIDE_CONFIG
        self.dilation_config = cfg.MODEL.ITD_BACKBONE.DILATION_CONFIG
        self.ksize_config = cfg.MODEL.ITD_BACKBONE.KSIZE_CONFIG

        for i in range(len(body_channels)):
            name = "block" + str(i)
            in_channels = body_channels[i][0]
            bottleneck_channels = body_channels[i][1]
            out_channels = body_channels[i][2]

            ### Construct blocks
            block = ITDBottleneckBlock(
                        in_channels=in_channels,
                        bottleneck_channels=bottleneck_channels,
                        out_channels=out_channels,
                        stride_options=stride_options,
                        dilation_options=dilation_options,
                        ksize_options=ksize_options,
                        norm=self.norm_func,
                    )

            self.add_module(name, block)
            self.block_names.append(name)
            self.return_feature_map[name] = return_features[i]

        if cfg.MODEL.ITD_BACKBONE.FREEZE_STEM:
            self.stem.freeze()


    def forward(self, x, config_combo):
        #print("input:", x.shape)
        outputs = []
        # Forward thru stem
        x = self.stem(x)
        #print("stem:", x.shape)
        curr_dynamic_stride_idx = 0
        curr_dynamic_dilation_idx = 0
        curr_dynamic_ksize_idx = 0
        for i, block_name in enumerate(self.block_names):
            #print("\ni:", i, block_name)
            # If layer is stride-dynamic
            if self.stride_config[i][0] == 1:
                curr_stride = config_combo[0][curr_dynamic_stride_idx]
                curr_dynamic_stride_idx += 1
            else:
                curr_stride = self.stride_config[i][1]
            # If layer is dilation-dynamic
            if self.dilation_config[i][0] == 1:
                curr_dilation = config_combo[1][curr_dynamic_dilation_idx]
                curr_dynamic_dilation_idx += 1
            else:
                curr_dilation = self.dilation_config[i][1]
            # If layer is ksize-dynamic
            if self.ksize_config[i][0] == 1:
                curr_ksize = config_combo[2][curr_dynamic_ksize_idx]
                curr_dynamic_ksize_idx += 1
            else:
                curr_ksize = self.ksize_config[i][1]
            
            #print("stride:", curr_stride)
            #print("dilation:", curr_dilation)
            #print("ksizes:", curr_ksize)
            x = getattr(self, block_name)(x, curr_stride, curr_dilation, curr_ksize)
            #print("x:", x.shape)
            if self.return_feature_map[block_name]:
                outputs.append(x)

        return outputs 



@BACKBONE_REGISTRY.register()
def build_itdresnet_backbone(cfg, input_shape):
    return ITDResNet(cfg, input_shape.channels)
