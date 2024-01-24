# Copyright (c) OpenMMLab. All rights reserved.
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
                 exit_indice=2,
                 freeze_backbone=True):
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

        self.exit_indice = exit_indice
        self.freeze_backbone = freeze_backbone
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
        
        # self.first_exit = SparseBasicBlock(
        #                         32,
        #                         32,
        #                         stride=2,
        #                         norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        #                         conv_cfg=dict(type='SubMConv3d'))
        if self.freeze_backbone:
            self._freeze_encoder()
        exit_indices_to_freeze = tuple([i for i in range(exit_indice)])
        self._freeze_encoder_exits(indices=exit_indices_to_freeze)

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
        layer_idx = 0

        exit_indice = self.exit_indice

        for encoder_layer in self.encoder_layers:
            # ts = time.time()
            # run encoder layer with backbone frozen
            x = encoder_layer(x) # x.dense().shape: -> [B, 32, 21, 512, 512] -> [B, 64, 11, 256, 256] -> [B, 128, 5, 128, 128] -> [B, 128, 5, 128, 128]
            
            if layer_idx < self.stage_num - 1:
                downsampler = self.encoder_downsamplers[layer_idx]
                if layer_idx <= exit_indice:
                    out = downsampler(x)
                else:
                    out = downsampler(early_exit_features[-1])
                early_exit_features.append(out)
            encode_features.append(x)
            layer_idx += 1

        early_exit_features.append(x)
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(early_exit_features[exit_indice])
        spatial_features = out.dense()

        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)

        return spatial_features

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

    def _freeze_encoder_exits(self, indices=(0, 1, 2)):
        # make sure that given indices does not exceed the number of encoder exits
        assert max(indices) < len(self.encoder_downsamplers)
        for i in indices:
            for param in self.encoder_downsamplers[i].parameters():
                param.requires_grad = False

# Copyright (c) OpenMMLab. All rights reserved.
# import torch
# from mmcv.ops import points_in_boxes_all, three_interpolate, three_nn
# from mmcv.runner import auto_fp16
# from torch import nn as nn

# from mmdet3d.ops import SparseBasicBlock, make_sparse_convmodule
# from mmdet3d.ops.spconv import IS_SPCONV2_AVAILABLE
# from mmdet.models.losses import sigmoid_focal_loss, smooth_l1_loss
# from mmdet3d.models.builder import MIDDLE_ENCODERS

# if IS_SPCONV2_AVAILABLE:
#     from spconv.pytorch import SparseConvTensor, SparseSequential
# else:
#     from mmcv.ops import SparseConvTensor, SparseSequential


# @MIDDLE_ENCODERS.register_module()
# class EarlyExitSparseEncoder(nn.Module):
#     r"""Sparse encoder for SECOND and Part-A2.

#     Args:
#         in_channels (int): The number of input channels.
#         sparse_shape (list[int]): The sparse shape of input tensor.
#         order (list[str], optional): Order of conv module.
#             Defaults to ('conv', 'norm', 'act').
#         norm_cfg (dict, optional): Config of normalization layer. Defaults to
#             dict(type='BN1d', eps=1e-3, momentum=0.01).
#         base_channels (int, optional): Out channels for conv_input layer.
#             Defaults to 16.
#         output_channels (int, optional): Out channels for conv_out layer.
#             Defaults to 128.
#         encoder_channels (tuple[tuple[int]], optional):
#             Convolutional channels of each encode block.
#             Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
#         encoder_paddings (tuple[tuple[int]], optional):
#             Paddings of each encode block.
#             Defaults to ((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)).
#         block_type (str, optional): Type of the block to use.
#             Defaults to 'conv_module'.
#     """

#     def __init__(self,
#                  in_channels,
#                  sparse_shape,
#                  order=('conv', 'norm', 'act'),
#                  norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
#                  base_channels=16,
#                  output_channels=128,
#                  encoder_channels=((16, ), (32, 32, 32), (64, 64, 64), (64, 64,
#                                                                         64)),
#                  encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1,
#                                                                  1)),
#                  block_type='conv_module'):
        
#         super().__init__()
#         assert block_type in ['conv_module', 'basicblock']
#         self.sparse_shape = sparse_shape
#         self.in_channels = in_channels
#         self.order = order
#         self.base_channels = base_channels
#         self.output_channels = output_channels
#         self.encoder_channels = encoder_channels
#         self.encoder_paddings = encoder_paddings
#         self.stage_num = len(self.encoder_channels)
#         self.fp16_enabled = False
#         # Spconv init all weight on its own

#         assert isinstance(order, tuple) and len(order) == 3
#         assert set(order) == {'conv', 'norm', 'act'}

#         if self.order[0] != 'conv':  # pre activate
#             self.conv_input = make_sparse_convmodule(
#                 in_channels,
#                 self.base_channels,
#                 3,
#                 norm_cfg=norm_cfg,
#                 padding=1,
#                 indice_key='subm1',
#                 conv_type='SubMConv3d',
#                 order=('conv', ))
#         else:  # post activate
#             self.conv_input = make_sparse_convmodule(
#                 in_channels,
#                 self.base_channels,
#                 3,
#                 norm_cfg=norm_cfg,
#                 padding=1,
#                 indice_key='subm1',
#                 conv_type='SubMConv3d')

#         encoder_out_channels = self.make_encoder_layers(
#             make_sparse_convmodule,
#             norm_cfg,
#             self.base_channels,
#             block_type=block_type)

#         self.conv_out = make_sparse_convmodule(
#             encoder_out_channels,
#             self.output_channels,
#             kernel_size=(3, 1, 1),
#             stride=(2, 1, 1),
#             norm_cfg=norm_cfg,
#             padding=0,
#             indice_key='spconv_down2',
#             conv_type='SparseConv3d')

#     @auto_fp16(apply_to=('voxel_features', ))
#     def forward(self, voxel_features, coors, batch_size):
#         """Forward of SparseEncoder.

#         Args:
#             voxel_features (torch.Tensor): Voxel features in shape (N, C).
#             coors (torch.Tensor): Coordinates in shape (N, 4),
#                 the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
#             batch_size (int): Batch size.

#         Returns:
#             dict: Backbone features.
#         """
#         coors = coors.int()
#         input_sp_tensor = SparseConvTensor(voxel_features, coors,
#                                            self.sparse_shape, batch_size)
#         x = self.conv_input(input_sp_tensor)

#         encode_features = []
#         for encoder_layer in self.encoder_layers:
#             x = encoder_layer(x)
#             encode_features.append(x)

#         # for detection head
#         # [200, 176, 5] -> [200, 176, 2]
#         out = self.conv_out(encode_features[-1])
#         spatial_features = out.dense()

#         N, C, D, H, W = spatial_features.shape
#         spatial_features = spatial_features.view(N, C * D, H, W)

#         return spatial_features

#     def make_encoder_layers(self,
#                             make_block,
#                             norm_cfg,
#                             in_channels,
#                             block_type='conv_module',
#                             conv_cfg=dict(type='SubMConv3d')):
#         """make encoder layers using sparse convs.

#         Args:
#             make_block (method): A bounded function to build blocks.
#             norm_cfg (dict[str]): Config of normalization layer.
#             in_channels (int): The number of encoder input channels.
#             block_type (str, optional): Type of the block to use.
#                 Defaults to 'conv_module'.
#             conv_cfg (dict, optional): Config of conv layer. Defaults to
#                 dict(type='SubMConv3d').

#         Returns:
#             int: The number of encoder output channels.
#         """
#         assert block_type in ['conv_module', 'basicblock']
#         self.encoder_layers = SparseSequential()

#         for i, blocks in enumerate(self.encoder_channels):
#             blocks_list = []
#             for j, out_channels in enumerate(tuple(blocks)):
#                 padding = tuple(self.encoder_paddings[i])[j]
#                 # each stage started with a spconv layer
#                 # except the first stage
#                 if i != 0 and j == 0 and block_type == 'conv_module':
#                     blocks_list.append(
#                         make_block(
#                             in_channels,
#                             out_channels,
#                             3,
#                             norm_cfg=norm_cfg,
#                             stride=2,
#                             padding=padding,
#                             indice_key=f'spconv{i + 1}',
#                             conv_type='SparseConv3d'))
#                 elif block_type == 'basicblock':
#                     if j == len(blocks) - 1 and i != len(
#                             self.encoder_channels) - 1:
#                         blocks_list.append(
#                             make_block(
#                                 in_channels,
#                                 out_channels,
#                                 3,
#                                 norm_cfg=norm_cfg,
#                                 stride=2,
#                                 padding=padding,
#                                 indice_key=f'spconv{i + 1}',
#                                 conv_type='SparseConv3d'))
#                     else:
#                         blocks_list.append(
#                             SparseBasicBlock(
#                                 out_channels,
#                                 out_channels,
#                                 norm_cfg=norm_cfg,
#                                 conv_cfg=conv_cfg))
#                 else:
#                     blocks_list.append(
#                         make_block(
#                             in_channels,
#                             out_channels,
#                             3,
#                             norm_cfg=norm_cfg,
#                             padding=padding,
#                             indice_key=f'subm{i + 1}',
#                             conv_type='SubMConv3d'))
#                 in_channels = out_channels
#             stage_name = f'encoder_layer{i + 1}'
#             stage_layers = SparseSequential(*blocks_list)
#             self.encoder_layers.add_module(stage_name, stage_layers)
#         return out_channels

