# encoding: utf-8

import functools
import torch.nn as nn
import numpy as np
import src.helper.torch_utils as torch_utils


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, drop_prob=0.4, n_blocks=6, padding_type='reflect', **kwargs):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super().__init__()
        use_dropout = False
        if drop_prob > 0:
            use_dropout = True

        final_activation_name = kwargs.get('final_activation', 'tanh')
        conv_layer_name = kwargs.get('conv_layer', 'conv2d')
        block_name = kwargs.get('block_name', 'resnetblock')
        n_downsampling = kwargs.get('downsampling', 2)
        block = torch_utils.block_selector(block_name)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 torch_utils.module_selector(conv_layer_name)(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [torch_utils.module_selector(conv_layer_name)(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks
            model += [
                block(in_chans=ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      # Added a 1x1 kernel to smoothen out the blocks from the Conv Transpose..(that is the idea)
                      # nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 2), kernel_size=1),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [torch_utils.module_selector(conv_layer_name)(ngf, output_nc, kernel_size=7, padding=0)]
        final_activation = torch_utils.activation_selector(final_activation_name)
        model += [final_activation]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

