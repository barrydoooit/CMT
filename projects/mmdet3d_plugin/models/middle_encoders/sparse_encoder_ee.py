# Copyright (c) OpenMMLab. All rights reserved.
import json
import random
import torch
from mmcv.ops import points_in_boxes_all, three_interpolate, three_nn
from mmcv.runner import auto_fp16
from torch import nn as nn

from mmdet3d.ops import SparseBasicBlock, make_sparse_convmodule
from mmdet3d.ops.spconv import IS_SPCONV2_AVAILABLE
from mmdet.models.losses import sigmoid_focal_loss, smooth_l1_loss
from mmdet3d.models.builder import MIDDLE_ENCODERS

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor, SparseSequential
else:
    from mmcv.ops import SparseConvTensor, SparseSequential
import time
from thop import profile



@MIDDLE_ENCODERS.register_module()
class EarlyExitSparseEncoder(nn.Module):
    r"""Sparse encoder for SECOND and Part-A2.

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        order (list[str], optional): Order of conv module.
            Defaults to ('conv', 'norm', 'act').
        norm_cfg (dict, optional): Config of normalization layer. Defaults to
            dict(type='BN1d', eps=1e-3, momentum=0.01).
        base_channels (int, optional): Out channels for conv_input layer.
            Defaults to 16.
        output_channels (int, optional): Out channels for conv_out layer.
            Defaults to 128.
        encoder_channels (tuple[tuple[int]], optional):
            Convolutional channels of each encode block.
            Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
        encoder_paddings (tuple[tuple[int]], optional):
            Paddings of each encode block.
            Defaults to ((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)).
        block_type (str, optional): Type of the block to use.
            Defaults to 'conv_module'.
    """

    def __init__(self,
                 in_channels,
                 sparse_shape,
                 order=('conv', 'norm', 'act'),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 base_channels=16,
                 output_channels=128,
                 encoder_channels=((16, ), (32, 32, 32), (64, 64, 64), (64, 64,
                                                                        64)),
                 encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1,
                                                                 1)),
                 block_type='conv_module',
                 exit_paddings=None,
                 test_exit_indice=1,
                 freeze_backbone=False):
        super().__init__()
        assert block_type in ['conv_module', 'basicblock']
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.stage_num = len(self.encoder_channels)
        self.fp16_enabled = False
        # Spconv init all weight on its own

        self.test_exit_indice = test_exit_indice
        self.freeze_backbone = freeze_backbone
        self.exit_paddings = exit_paddings
        self.train_exit_sequence = (3, 0, 1, 2)
        self.cur_train_exit_idx = 0

        self.flops_records = []

        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {'conv', 'norm', 'act'}

        if self.order[0] != 'conv':  # pre activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d',
                order=('conv', ))
        else:  # post activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d')

        encoder_out_channels = self.make_encoder_layers(
            make_sparse_convmodule,
            norm_cfg,
            self.base_channels,
            block_type=block_type)

        encoder_early_exit_out_channels = self.make_encoder_layer_downsampler(
            make_sparse_convmodule,
            norm_cfg,
            block_type=block_type)

        self.conv_out = make_sparse_convmodule(
            encoder_out_channels,
            self.output_channels,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
            norm_cfg=norm_cfg,
            padding=0,
            indice_key='spconv_down2',
            conv_type='SparseConv3d')
        
        # self.first_exit = SparseBasicBlock(
        #                         32,
        #                         32,
        #                         stride=2,
        #                         norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        #                         conv_cfg=dict(type='SubMConv3d'))
        if self.freeze_backbone:
            self._freeze_encoder()

        # exit_indices_to_freeze = tuple([i for i in range(exit_indice)])
        # self._freeze_encoder_exits(indices=exit_indices_to_freeze)

    @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features, coors, batch_size):
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, C).
            coors (torch.Tensor): Coordinates in shape (N, 4),
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            dict: Backbone features.
        """
        coors = coors.int()
        input_sp_tensor = SparseConvTensor(voxel_features, coors,
                                           self.sparse_shape, batch_size)
        # ts = time.time()
        macs_dict = {}
        # macs, params = profile(self.conv_input, inputs=(input_sp_tensor, ))
        # macs_dict['conv_input'] = macs
        x = self.conv_input(input_sp_tensor)
        # print(f"Conv input time: {time.time() - ts}")
        # x.dense().shape: [B, 16, 41, 1024, 1024]
        encode_features = []
        early_exit_features = []
        layer_idx = 0

        if self.training:
            exit_indice_train = self.train_exit_sequence[self.cur_train_exit_idx]
            self.cur_train_exit_idx = (self.cur_train_exit_idx + 1) % len(self.train_exit_sequence)
            for encoder_layer in self.encoder_layers:
                
                if layer_idx > exit_indice_train:
                    with torch.no_grad():
                        x = encoder_layer(x)
                else:
                    x = encoder_layer(x) # x.dense().shape: -> [B, 32, 21, 512, 512] -> [B, 64, 11, 256, 256] -> [B, 128, 5, 128, 128] -> [B, 128, 5, 128, 128]
                
                if layer_idx < self.stage_num - 1:
                    downsampler = self.encoder_downsamplers[layer_idx]
                    if layer_idx < exit_indice_train:
                        with torch.no_grad():
                            out = downsampler(x)
                    elif layer_idx == exit_indice_train:
                        out = downsampler(x)
                    else:
                        out = downsampler(early_exit_features[-1])
                    early_exit_features.append(out)
                encode_features.append(x)
                layer_idx += 1
            
            if exit_indice_train == self.stage_num - 1: # last layer
                out = self.conv_out(encode_features[-1])
            else:
                out = self.conv_out(early_exit_features[-1])
        else:
            exit_indice_test = self.test_exit_indice
            to_break = False
            for encoder_layer in self.encoder_layers:
                # macs, params = profile(encoder_layer, inputs=(x, ))
                # macs_dict[f'encoder_layer{layer_idx}'] = macs
                x = encoder_layer(x)
                if layer_idx == exit_indice_test:
                    to_break = True
                    if exit_indice_test == self.stage_num - 1: # last layer
                        early_exit_features.append(x)
                    else:
                        for downsampler in self.encoder_downsamplers[layer_idx:]:
                            x = downsampler(x)
                            early_exit_features.append(x)
                if to_break:
                    break    
                layer_idx += 1
            # macs, params = profile(self.conv_out, inputs=(early_exit_features[-1], ))
            # macs_dict[f'conv_out'] = macs
            out = self.conv_out(early_exit_features[-1])
            # with open('/workspace/work_dirs/temp/ecndoer_macs.json', 'a') as f:
            #     json.dump(macs_dict, f)
            #     f.write('\n')
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        
        spatial_features = out.dense()
        
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)

        features_out: dict = {'student': spatial_features}
        # if self.training:
        #     # get original final spatial feature into conv_out without gradient
        #     assert len(encode_features) == self.stage_num
        #     with torch.no_grad():
        #         teacher_out = self.conv_out(encode_features[-1])
        #         spatial_features_teacher = teacher_out.dense()
        #         spatial_features_teacher = spatial_features_teacher.view(N, C * D, H, W)
        #     features_out['teacher'] = spatial_features_teacher
        return features_out

    def make_encoder_layers(self,
                            make_block,
                            norm_cfg,
                            in_channels,
                            block_type='conv_module',
                            conv_cfg=dict(type='SubMConv3d')):
        """make encoder layers using sparse convs.

        Args:
            make_block (method): A bounded function to build blocks.
            norm_cfg (dict[str]): Config of normalization layer.
            in_channels (int): The number of encoder input channels.
            block_type (str, optional): Type of the block to use.
                Defaults to 'conv_module'.
            conv_cfg (dict, optional): Config of conv layer. Defaults to
                dict(type='SubMConv3d').

        Returns:
            int: The number of encoder output channels.
        """
        assert block_type in ['conv_module', 'basicblock']
        # self.encoder_layers = SparseSequential()
        self.encoder_layers = nn.ModuleList()
        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                # each stage started with a spconv layer
                # except the first stage
                if i != 0 and j == 0 and block_type == 'conv_module':
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            stride=2,
                            padding=padding,
                            indice_key=f'spconv{i + 1}',
                            conv_type='SparseConv3d'))
                elif block_type == 'basicblock':
                    if j == len(blocks) - 1 and i != len(
                            self.encoder_channels) - 1:
                        blocks_list.append(
                            make_block(
                                in_channels,
                                out_channels,
                                3,
                                norm_cfg=norm_cfg,
                                stride=2,
                                padding=padding,
                                indice_key=f'spconv{i + 1}',
                                conv_type='SparseConv3d'))
                    else:
                        blocks_list.append(
                            SparseBasicBlock(
                                out_channels,
                                out_channels,
                                norm_cfg=norm_cfg,
                                conv_cfg=conv_cfg))
                else:
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            padding=padding,
                            indice_key=f'subm{i + 1}',
                            conv_type='SubMConv3d'))
                in_channels = out_channels

            stage_name = f'encoder_layer{i + 1}'
            stage_layers = SparseSequential(*blocks_list)
            # self.encoder_layers.add_module(stage_name, stage_layers)
            self.encoder_layers.append(stage_layers)

            # stage_layers = SparseSequential()
            # for k, block in enumerate(blocks_list):
            #     stage_layers.add_module(f'encoder_layer{i + 1}_{k}', block)
            # self.encoder_layers.append(stage_layers)
        return out_channels
    
    def make_encoder_layer_downsampler(self,
                            make_block,
                            norm_cfg,
                            block_type='conv_module',
                            conv_cfg=dict(type='SubMConv3d')):
        """make downsampler for encoder layers using sparse convs.

        Args:
            make_block (method): A bounded function to build blocks.
            norm_cfg (dict[str]): Config of normalization layer.
            block_type (str, optional): Type of the block to use.
                Defaults to 'conv_module'.
            conv_cfg (dict, optional): Config of conv layer. Defaults to
                dict(type='SubMConv3d').

        Returns:
            int: The number of encoder output channels.
        """
        if block_type != 'basicblock':
            raise NotImplementedError("[SparseEncoder] Only support basicblock")
        
        # self.encoder_downsamplers = SparseSequential()
        self.encoder_downsamplers = nn.ModuleList()


        for i, blocks in enumerate(self.encoder_channels[:-1]):
            blocks_list = []
            j = len(blocks) - 1
            layer_out_channel = tuple(blocks)[j]
            next_layer_out_channel = tuple(self.encoder_channels[i+1])[-1]
            stride = 2 if i != len(self.encoder_channels) - 2 else 1

            blocks_list.append(
            #     SparseBasicBlock(
            #                     layer_out_channel,
            #                     layer_out_channel,
            #                     stride=stride,
            #                     norm_cfg=norm_cfg,
            #                     conv_cfg=conv_cfg)
            # )
                make_block(
                    layer_out_channel,
                    layer_out_channel,
                    3,
                    norm_cfg=norm_cfg,
                    stride=stride,
                    padding=self.exit_paddings[i],
                    indice_key=f'exit_ds{i + 1}',
                    conv_type='SparseConv3d'))
            # add 1x1 conv to increase channels
            blocks_list.append(
                make_block(
                    layer_out_channel,
                    next_layer_out_channel,
                    1,
                    norm_cfg=norm_cfg,
                    stride=1,
                    padding=0,
                    indice_key=f'exit_ce{i + 1}',
                    conv_type='SparseConv3d'))
            # stage_name = f'encoder_exit{i + 1}' # layer name
            # stage_layers = SparseSequential()
            # for k, block in enumerate(blocks_list):
            #     stage_layers.add_module(f'encoder_exit{i + 1}_{k}', block)
            # self.encoder_downsamplers.append(stage_layers)
            # # self.encoder_downsamplers.add_module(f'encoder_exit{i + 1}', stage_layers)

            stage_layers = SparseSequential(*blocks_list)
            self.encoder_downsamplers.append(stage_layers)

        return next_layer_out_channel
    
    def _freeze_encoder(self):
        for param in self.conv_input.parameters():
            param.requires_grad = False
        for param in self.encoder_layers.parameters():
            param.requires_grad = False
        for param in self.conv_out.parameters():
            param.requires_grad = False

    def _freeze_encoder_exits(self, indices=(0, 1, 2)):
        # make sure that given indices does not exceed the number of encoder exits
        assert max(indices) < len(self.encoder_downsamplers)
        for i in indices:
            for param in self.encoder_downsamplers[i].parameters():
                param.requires_grad = False

@MIDDLE_ENCODERS.register_module()
class EarlyExitSparseEncoderV2(nn.Module):
    r"""Sparse encoder for SECOND and Part-A2.

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        order (list[str], optional): Order of conv module.
            Defaults to ('conv', 'norm', 'act').
        norm_cfg (dict, optional): Config of normalization layer. Defaults to
            dict(type='BN1d', eps=1e-3, momentum=0.01).
        base_channels (int, optional): Out channels for conv_input layer.
            Defaults to 16.
        output_channels (int, optional): Out channels for conv_out layer.
            Defaults to 128.
        encoder_channels (tuple[tuple[int]], optional):
            Convolutional channels of each encode block.
            Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
        encoder_paddings (tuple[tuple[int]], optional):
            Paddings of each encode block.
            Defaults to ((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)).
        block_type (str, optional): Type of the block to use.
            Defaults to 'conv_module'.
    """

    def __init__(self,
                 in_channels,
                 sparse_shape,
                 order=('conv', 'norm', 'act'),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 base_channels=16,
                 output_channels=128,
                 encoder_channels=((16, ), (32, 32, 32), (64, 64, 64), (64, 64,
                                                                        64)),
                 encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1,
                                                                 1)),
                 block_type='conv_module',
                 exit_paddings=(2, [1, 2, 2], 1),
                 test_exit_indice=1):
        super().__init__()
        assert block_type in ['conv_module', 'basicblock']
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.stage_num = len(self.encoder_channels)
        self.fp16_enabled = False
        # Spconv init all weight on its own

        self.test_exit_indice = test_exit_indice
        self.exit_paddings = exit_paddings

        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {'conv', 'norm', 'act'}

        if self.order[0] != 'conv':  # pre activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d',
                order=('conv', ))
        else:  # post activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d')

        encoder_out_channels = self.make_encoder_layers(
            make_sparse_convmodule,
            norm_cfg,
            self.base_channels,
            block_type=block_type)

        encoder_early_exit_out_channels = self.make_encoder_layer_downsampler(
            make_sparse_convmodule,
            norm_cfg,
            block_type=block_type)

        self.conv_out = make_sparse_convmodule(
            encoder_out_channels,
            self.output_channels,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
            norm_cfg=norm_cfg,
            padding=0,
            indice_key='spconv_down2',
            conv_type='SparseConv3d')
        
    @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features, coors, batch_size):
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, C).
            coors (torch.Tensor): Coordinates in shape (N, 4),
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            dict: Backbone features.
        """
        coors = coors.int()
        input_sp_tensor = SparseConvTensor(voxel_features, coors,
                                           self.sparse_shape, batch_size)
        # ts = time.time()
        x = self.conv_input(input_sp_tensor)
        # print(f"Conv input time: {time.time() - ts}")
        # x.dense().shape: [B, 16, 41, 1024, 1024]
        encode_features = []
        early_exit_features = []
        out_features = []
        

        layer_idx = 0
        if self.training:
            for encoder_layer in self.encoder_layers:
                x = encoder_layer(x) # x.dense().shape: -> [B, 32, 21, 512, 512] -> [B, 64, 11, 256, 256] -> [B, 128, 5, 128, 128] -> [B, 128, 5, 128, 128]
                if layer_idx == self.stage_num - 1:
                    out = self.conv_out(x)
                else:
                    exit_x = self.encoder_downsamplers[layer_idx](x)
                    out = self.conv_out(exit_x)
                out_features.append(out)
                layer_idx += 1
        else:
            exit_indice_test = self.test_exit_indice
            for encoder_layer in self.encoder_layers:
                x = encoder_layer(x)
                if layer_idx == self.stage_num - 1:
                    out = self.conv_out(x)
                    out_features.append(out)
                elif layer_idx == exit_indice_test:
                    exit_x = self.encoder_downsamplers[layer_idx](x)
                    out = self.conv_out(exit_x)
                    out_features.append(out)
                    break
                layer_idx += 1
            

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        
        spatial_features = [o.dense() for o in out_features]
        
        N, C, D, H, W = spatial_features[0].shape
        spatial_features = [sf.view(N, C * D, H, W) for sf in spatial_features]

        features_out: dict = {'student': spatial_features}
        return features_out

    def make_encoder_layers(self,
                            make_block,
                            norm_cfg,
                            in_channels,
                            block_type='conv_module',
                            conv_cfg=dict(type='SubMConv3d')):
        """make encoder layers using sparse convs.

        Args:
            make_block (method): A bounded function to build blocks.
            norm_cfg (dict[str]): Config of normalization layer.
            in_channels (int): The number of encoder input channels.
            block_type (str, optional): Type of the block to use.
                Defaults to 'conv_module'.
            conv_cfg (dict, optional): Config of conv layer. Defaults to
                dict(type='SubMConv3d').

        Returns:
            int: The number of encoder output channels.
        """
        assert block_type in ['conv_module', 'basicblock']
        # self.encoder_layers = SparseSequential()
        self.encoder_layers = nn.ModuleList()
        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                # each stage started with a spconv layer
                # except the first stage
                if i != 0 and j == 0 and block_type == 'conv_module':
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            stride=2,
                            padding=padding,
                            indice_key=f'spconv{i + 1}',
                            conv_type='SparseConv3d'))
                elif block_type == 'basicblock':
                    if j == len(blocks) - 1 and i != len(
                            self.encoder_channels) - 1:
                        blocks_list.append(
                            make_block(
                                in_channels,
                                out_channels,
                                3,
                                norm_cfg=norm_cfg,
                                stride=2,
                                padding=padding,
                                indice_key=f'spconv{i + 1}',
                                conv_type='SparseConv3d'))
                    else:
                        blocks_list.append(
                            SparseBasicBlock(
                                out_channels,
                                out_channels,
                                norm_cfg=norm_cfg,
                                conv_cfg=conv_cfg))
                else:
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            padding=padding,
                            indice_key=f'subm{i + 1}',
                            conv_type='SubMConv3d'))
                in_channels = out_channels

            stage_name = f'encoder_layer{i + 1}'
            stage_layers = SparseSequential(*blocks_list)
            # self.encoder_layers.add_module(stage_name, stage_layers)
            self.encoder_layers.append(stage_layers)

            # stage_layers = SparseSequential()
            # for k, block in enumerate(blocks_list):
            #     stage_layers.add_module(f'encoder_layer{i + 1}_{k}', block)
            # self.encoder_layers.append(stage_layers)
        return out_channels
    
    def make_encoder_layer_downsampler(self,
                            make_block,
                            norm_cfg,
                            block_type='basicblock',
                            conv_cfg=dict(type='SubMConv3d')):
        """make downsampler for encoder layers using sparse convs.

        Args:
            make_block (method): A bounded function to build blocks.
            norm_cfg (dict[str]): Config of normalization layer.
            block_type (str, optional): Type of the block to use.
                Defaults to 'conv_module'.
            conv_cfg (dict, optional): Config of conv layer. Defaults to
                dict(type='SubMConv3d').

        Returns:
            int: The number of encoder output channels.
        """
        if block_type != 'basicblock':
            raise NotImplementedError("[SparseEncoder] Only support basicblock")
        
        self.encoder_downsamplers = nn.ModuleList()

        layer_out_channels = [tuple(block)[-1] for block in self.encoder_channels]
        strides = [4, 2, 1, 1]
        kernel_sizes = [7, 5, 3, 3]
        final_out_channel = layer_out_channels[-1]

        for i in range(len(self.encoder_channels) - 1):
            blocks_list  = []
            downsampler_in_channels = layer_out_channels[i]
            blocks_list.append(
                make_block(
                    downsampler_in_channels,
                    downsampler_in_channels,
                    kernel_sizes[i],
                    norm_cfg=norm_cfg,
                    stride=strides[i],
                    padding=self.exit_paddings[i],
                    indice_key=f'exit_{i + 1}_conv',
                    conv_type='SparseConv3d'))
            # add 1x1 conv to increase channels
            blocks_list.append(
                make_block(
                    downsampler_in_channels,
                    final_out_channel,
                    1,
                    norm_cfg=norm_cfg,
                    stride=1,
                    padding=0,
                    indice_key=f'exit_{i + 1}_lift',
                    conv_type='SparseConv3d'))

            stage_layers = SparseSequential(*blocks_list)
            self.encoder_downsamplers.append(stage_layers)

        return final_out_channel
    