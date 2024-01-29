# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import mmcv
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmcv.runner import force_fp32, auto_fp16
from mmdet.core import multi_apply
from mmdet.models import DETECTORS
from mmdet.models.builder import build_backbone
from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin import SPConvVoxelization

from projects.utils import ModuleStartEndTimeTracker

@DETECTORS.register_module()
class EarlyExitCmtDetectorV2(MVXTwoStageDetector):

    def __init__(self,
                 use_grid_mask=False,
                 log=True,
                 img_frozen=False,
                 voxel_frozen=False,
                 pts_frozen=False,
                 bbox_head_frozen=False,
                 **kwargs):
        pts_voxel_cfg = kwargs.get('pts_voxel_layer', None)
        kwargs['pts_voxel_layer'] = None
        super(EarlyExitCmtDetectorV2, self).__init__(**kwargs)
        
        self.use_grid_mask = use_grid_mask
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        if pts_voxel_cfg:
            self.pts_voxel_layer = SPConvVoxelization(**pts_voxel_cfg)
        
        # self.module_time_tracker = ModuleStartEndTimeTracker('/workspace/work_dirs/temp/time_vov_fusion.json', is_tracking=False)
        if img_frozen:
            self._freeze_image_branch()
        if voxel_frozen:
            self._freeze_voxelization()
        if pts_frozen:
            self._freeze_pts_after_encoder()
        if bbox_head_frozen:
            self._freeze_pts_bbox_head()


    def init_weights(self):
        """Initialize model weights."""
        super(EarlyExitCmtDetectorV2, self).init_weights()

    @auto_fp16(apply_to=('img'), out_fp32=True) 
    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            # self.module_time_tracker.register_start('img_backbone')
            img_feats = self.img_backbone(img.float())
            # ts_end_img_backbone = self.module_time_tracker.register_end('img_backbone')
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        # self.module_time_tracker.register_start('img_neck', ts_end_img_backbone)
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        # self.module_time_tracker.register_end('img_neck')
        return img_feats

    @force_fp32(apply_to=('pts', 'img_feats'))
    def extract_pts_feat(self, pts, img_feats, img_metas):
        # self.module_time_tracker.register_start('pts_voxelize')
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        if pts is None:
            return None
        voxels, num_points, coors = self.voxelize(pts)

        # ts_end_pts_voxelize =  self.module_time_tracker.register_end('pts_voxelize')
        # self.module_time_tracker.register_start('pts_voxel_encoder', ts_end_pts_voxelize)
        ###
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                )
        batch_size = coors[-1, 0] + 1

        # ts_end_pts_voxel_encoder = self.module_time_tracker.register_end('pts_voxel_encoder')
        # self.module_time_tracker.register_start('pts_middle_encoder', ts_end_pts_voxel_encoder)
        ###
        teacher_student_features: dict = self.pts_middle_encoder(voxel_features, coors, batch_size)
        _x_list = teacher_student_features['student']
        
        # ts_end_pts_middle_encoder = self.module_time_tracker.register_end('pts_middle_encoder')
        # self.module_time_tracker.register_start('pts_backbone', ts_end_pts_middle_encoder)
        ###
        _x_list = [self.pts_backbone(_x) for _x in _x_list]

        # ts_end_pts_backbone = self.module_time_tracker.register_end('pts_backbone')
        # self.module_time_tracker.register_start('pts_neck', ts_end_pts_backbone)
        ###
        if self.with_pts_neck:
            _x_list = [self.pts_neck(_x) for _x in _x_list]

        # self.module_time_tracker.register_end('pts_neck')

        features_out = {'student': _x_list}
        # if self.training:
        #     with torch.no_grad():
        #         features_out['teacher'] = self.extract_pts_feat_post_middle_encoder(teacher_student_features['teacher'])
        #     return features_out
        return features_out

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        return (img_feats, pts_feats)
    
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """

        img_feats, pts_feats_dict = self.extract_feat(
            points, img=img, img_metas=img_metas)
        
        pts_feats = pts_feats_dict['student']

        losses = dict()
        assert len(pts_feats) == 4 or len(pts_feats) == 1
        losses_weight = (0.15, 0.2, 0.25, 0.3) if len(pts_feats) == 4 else (1.0,)
        for i, pts_feat in enumerate(pts_feats):
            losses_pts = self.forward_pts_train(pts_feat, img_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            for k, v in losses_pts.items():
                if k not in losses:
                    losses[k] = v * losses_weight[i]
                else:
                    losses[k] += v * losses_weight[i]
        return losses

    @force_fp32(apply_to=('pts_feats', 'img_feats'))
    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        if pts_feats is None:
            pts_feats = [None]
        if img_feats is None:
            img_feats = [None]
        outs = self.pts_bbox_head(pts_feats, img_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img=None, **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        if points is None:
            points = [None]
        if img is None:
            img = [None]
        for var, name in [(points, 'points'), (img, 'img'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        return self.simple_test(points[0], img_metas[0], img[0], **kwargs)
    
    @force_fp32(apply_to=('x', 'x_img'))
    def simple_test_pts(self, x, x_img, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, x_img, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ] 
        return bbox_results

    def simple_test(self, points, img_metas, img=None, rescale=False):
        # ts_start_total = self.module_time_tracker.register_start('total')
        # self.module_time_tracker.register_start('all_feat', ts_start_total)
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        if pts_feats is None:
            pts_feats = [None]
        if img_feats is None:
            img_feats = [None]
        
        # ts_end_all_feat = self.module_time_tracker.register_end('all_feat')

        # ts_start_bbox_all = self.module_time_tracker.register_start('bbox_all', ts_end_all_feat)
        # self.module_time_tracker.register_start('bbox_pts', ts_start_bbox_all)
        bbox_list = [dict() for i in range(len(img_metas))]
        if (pts_feats or img_feats) and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                pts_feats, img_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        # ts_end_bbox_pts = self.module_time_tracker.register_end('bbox_pts')
        # self.module_time_tracker.register_start('bbox_img', ts_end_bbox_pts)
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        # ts_end_bbox_img = self.module_time_tracker.register_end('bbox_img')
        # self.module_time_tracker.register_end('bbox_all', ts_end_bbox_img) 
        # self.module_time_tracker.register_end('total', ts_end_bbox_img)
        # self.module_time_tracker.dump()
        return bbox_list

    def _freeze_image_branch(self):
        """Freeze image branch during training."""
        if self.with_img_backbone:
            for m in self.img_backbone.modules():
                for param in m.parameters():
                    param.requires_grad = False
        if self.with_img_neck:
            for m in self.img_neck.modules():
                for param in m.parameters():
                    param.requires_grad = False
    
    def _freeze_voxelization(self):
        """Freeze voxelization during training."""
        for p in self.pts_voxel_layer.parameters():
            p.requires_grad = False
        for p in self.pts_voxel_encoder.parameters():
            p.requires_grad = False

    def _freeze_pts_after_encoder(self):
        """Freeze point cloud branch after voxel encoder during training."""
        for p in self.pts_backbone.parameters():
            p.requires_grad = False
        for p in self.pts_neck.parameters():
            p.requires_grad = False

    def _freeze_pts_bbox_head(self):
        """Freeze point cloud bbox head during training."""
        for p in self.pts_bbox_head.parameters():
            p.requires_grad = False