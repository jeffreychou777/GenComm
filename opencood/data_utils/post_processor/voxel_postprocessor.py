# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
3D Anchor Generator for Voxel
"""
import math
import sys

import numpy as np
import torch
from torch.nn.functional import sigmoid
import torch.nn.functional as F
from scipy.special import softmax
from opencood.data_utils.post_processor.base_postprocessor \
    import BasePostprocessor
from opencood.utils import box_utils
from opencood.utils.box_overlaps import bbox_overlaps
from opencood.visualization import vis_utils
from opencood.utils.common_utils import limit_period


class VoxelPostprocessor(BasePostprocessor):
    def __init__(self, anchor_params, train, class_names=None):
        super(VoxelPostprocessor, self).__init__(anchor_params, train, class_names=None)
        self.anchor_num = self.params['anchor_args']['num']
        self.max_num = self.params['max_num']
        self.aggregation = getattr(self.params, "aggregation", "nms")
        
        if class_names is not None:
            anchor_generator_config = anchor_params['anchor_args'][
                'anchor_generator_config']
            self.order = anchor_params['order']
            self.anchor_generator_config = anchor_generator_config
            # The order in the list will correspond to the order of class names in anchor_generator_config
            self.anchor_sizes = [config['anchor_sizes'] for config in
                                    anchor_generator_config]
            self.anchor_rotations = [config['anchor_rotations'] for config in
                                        anchor_generator_config]
            self.anchor_heights = [config['anchor_bottom_heights'] for config in
                                    anchor_generator_config]
            self.align_center = [config.get('align_center', False) for config in
                                    anchor_generator_config]
            self.anchor_class_names = [config['class_name'] for config in
                                        anchor_generator_config]
            self.matched_thresholds = {}
            self.unmatched_thresholds = {}
            for config in anchor_generator_config:
                self.matched_thresholds[config['class_name']] = config[
                    'matched_threshold']
                self.unmatched_thresholds[config['class_name']] = config[
                    'unmatched_threshold']


            assert len(self.anchor_sizes) == len(self.anchor_rotations) == len(
                self.anchor_heights)
            # Need to update all the anchor number
            self.num_of_anchor_sets = len(self.anchor_sizes)

            W = anchor_params['anchor_args']['W']  # x-axis
            H = anchor_params['anchor_args']['H']  # y-axis
            self.grid_size = np.array([W, H])
            # [x_min, y_min, z_min, x_max, y_max, z_max]
            self.cav_lidar_range = anchor_params['anchor_args']['cav_lidar_range']

    def generate_anchor_box(self):
        # load_voxel_params and load_point_pillar_params leads to the same anchor
        # if voxel_size * feature stride is the same.
        W = self.params['anchor_args']['W']
        H = self.params['anchor_args']['H']

        l = self.params['anchor_args']['l']
        w = self.params['anchor_args']['w']
        h = self.params['anchor_args']['h']
        r = self.params['anchor_args']['r']

        assert self.anchor_num == len(r)
        r = [math.radians(ele) for ele in r]

        vh = self.params['anchor_args']['vh'] # voxel_size
        vw = self.params['anchor_args']['vw']

        xrange = [self.params['anchor_args']['cav_lidar_range'][0],
                  self.params['anchor_args']['cav_lidar_range'][3]]
        yrange = [self.params['anchor_args']['cav_lidar_range'][1],
                  self.params['anchor_args']['cav_lidar_range'][4]]

        if 'feature_stride' in self.params['anchor_args']:
            feature_stride = self.params['anchor_args']['feature_stride']
        else:
            feature_stride = 2


        x = np.linspace(xrange[0] + vw, xrange[1] - vw, W // feature_stride) # vw is not precise, vw * feature_stride / 2 should be better?
        y = np.linspace(yrange[0] + vh, yrange[1] - vh, H // feature_stride)


        cx, cy = np.meshgrid(x, y)
        cx = np.tile(cx[..., np.newaxis], self.anchor_num) # center
        cy = np.tile(cy[..., np.newaxis], self.anchor_num)
        cz = np.ones_like(cx) * -1.0

        w = np.ones_like(cx) * w
        l = np.ones_like(cx) * l
        h = np.ones_like(cx) * h

        r_ = np.ones_like(cx)
        for i in range(self.anchor_num):
            r_[..., i] = r[i]

        if self.params['order'] == 'hwl': # pointpillar
            anchors = np.stack([cx, cy, cz, h, w, l, r_], axis=-1) # (50, 176, 2, 7)

        elif self.params['order'] == 'lhw':
            anchors = np.stack([cx, cy, cz, l, h, w, r_], axis=-1)
        else:
            sys.exit('Unknown bbx order.')

        return anchors

    def generate_anchor_box_v2xreal(self):
        grid_sizes = [self.grid_size[:2] // config['feature_map_stride'] for
                      config in self.anchor_generator_config]
        all_anchors = []
        num_anchors_per_location = []
        for grid_size, anchor_size, anchor_rotation, anchor_height, align_center in zip(
                grid_sizes, self.anchor_sizes, self.anchor_rotations,
                self.anchor_heights, self.align_center):
            num_anchors_per_location.append(
                len(anchor_rotation) * len(anchor_size) * len(anchor_height))
            if align_center:
                x_stride = (self.cav_lidar_range[3] - self.cav_lidar_range[
                    0]) / grid_size[0]
                y_stride = (self.cav_lidar_range[4] - self.cav_lidar_range[
                    1]) / grid_size[1]
                x_offset, y_offset = x_stride / 2, y_stride / 2
            else:
                x_stride = (self.cav_lidar_range[3] - self.cav_lidar_range[
                    0]) / (grid_size[0] - 1)
                y_stride = (self.cav_lidar_range[4] - self.cav_lidar_range[
                    1]) / (grid_size[1] - 1)
                x_offset, y_offset = 0, 0

            # (grid_sizes[1], )
            x_shifts = np.arange(self.cav_lidar_range[0] + x_offset,
                                 self.cav_lidar_range[3] + 1e-5, step=x_stride)
            # (grid_sizes[0], )
            y_shifts = np.arange(self.cav_lidar_range[1] + y_offset,
                                 self.cav_lidar_range[4] + 1e-5, step=y_stride)
            z_shifts = np.array(anchor_height)

            num_anchor_size = len(anchor_size)
            num_anchor_rotation = len(anchor_rotation)

            anchor_rotation = np.array(anchor_rotation)
            anchor_size = np.array(anchor_size)
            # [x_grid, y_grid, z_grid] indexing may need double check
            x_shifts, y_shifts, z_shifts = np.meshgrid(x_shifts, y_shifts,
                                                       z_shifts)
            anchors = np.concatenate([x_shifts, y_shifts, z_shifts], axis=-1)

            # (x_grid,y_grid, 3) -- l w h
            anchor_size = np.tile(anchor_size.reshape(1, -1, 3),
                                  (*anchors.shape[0:2], 1))

            if self.order == 'hwl':
                anchor_size = anchor_size[..., [2, 1, 0]]
            elif self.order == 'lhw':
                anchor_size = anchor_size[..., [0, 2, 1]]
            else:
                sys.exit('Unknown bbx order.')
            # (x_grid, y_grid, 6) -- [x, y, z, h, w, l]
            anchors = np.concatenate((anchors, anchor_size), axis=-1)
            # (x_grid, y_grid, 2, 6)
            anchors = np.tile(anchors[:, :, None, :],
                              (1, 1, num_anchor_rotation, 1))
            anchor_rotation = np.tile(anchor_rotation.reshape(1, 1, -1, 1), (
            *anchors.shape[0:2], num_anchor_size, 1))
            # (x_grid, y_grid, 2, 7) -- [x, y, z, h, w, l, yaw]
            anchors = np.concatenate([anchors, anchor_rotation], axis=-1)
            # # If shifted to box centers; disabled now
            # anchors[..., 2] += anchors[..., 5] / 2  # shift to box centers
            all_anchors.append(anchors)
        return all_anchors, num_anchors_per_location

    def generate_label(self, **kwargs):
        """
        Generate targets for training.

        Parameters
        ----------
        argv : list
            gt_box_center:(max_num, 7), anchor:(H, W, anchor_num, 7)

        Returns
        -------
        label_dict : dict
            Dictionary that contains all target related info.
        """
        assert self.params['order'] == 'hwl', 'Currently Voxel only support' \
                                              'hwl bbx order.'
        # (max_num, 7)
        gt_box_center = kwargs['gt_box_center']
        # (H, W, anchor_num, 7)
        anchors = kwargs['anchors']
        # (max_num)
        masks = kwargs['mask']

        # (H, W)
        feature_map_shape = anchors.shape[:2]

        # (H*W*anchor_num, 7)
        anchors = anchors.reshape(-1, 7)
        # normalization factor, (H * W * anchor_num)
        anchors_d = np.sqrt(anchors[:, 4] ** 2 + anchors[:, 5] ** 2)

        # (H, W, 2)
        pos_equal_one = np.zeros((*feature_map_shape, self.anchor_num))
        neg_equal_one = np.zeros((*feature_map_shape, self.anchor_num))
        # (H, W, self.anchor_num * 7)
        targets = np.zeros((*feature_map_shape, self.anchor_num * 7))

        # (n, 7)
        gt_box_center_valid = gt_box_center[masks == 1]
        # (n, 8, 3)
        gt_box_corner_valid = \
            box_utils.boxes_to_corners_3d(gt_box_center_valid,
                                          self.params['order'])
        # (H*W*anchor_num, 8, 3)
        anchors_corner = \
            box_utils.boxes_to_corners_3d(anchors,
                                          order=self.params['order'])
        # (H*W*anchor_num, 4)
        anchors_standup_2d = \
            box_utils.corner2d_to_standup_box(anchors_corner)
        # (n, 4)
        gt_standup_2d = \
            box_utils.corner2d_to_standup_box(gt_box_corner_valid)

        # (H*W*anchor_n)
        iou = bbox_overlaps(
            np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
            np.ascontiguousarray(gt_standup_2d).astype(np.float32),
        )

        # the anchor boxes has the largest iou across
        # shape: (n)
        id_highest = np.argmax(iou.T, axis=1)
        # [0, 1, 2, ..., n-1]
        id_highest_gt = np.arange(iou.T.shape[0])
        # make sure all highest iou is larger than 0
        mask = iou.T[id_highest_gt, id_highest] > 0
        id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]


        # find anchors iou > params['pos_iou']
        id_pos, id_pos_gt = \
            np.where(iou >
                     self.params['target_args']['pos_threshold'])
        #  find anchors iou  params['neg_iou']
        id_neg = np.where(np.sum(iou <
                                 self.params['target_args']['neg_threshold'],
                                 axis=1) == iou.shape[1])[0]
        id_pos = np.concatenate([id_pos, id_highest])
        id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])
        id_pos, index = np.unique(id_pos, return_index=True)
        id_pos_gt = id_pos_gt[index]
        id_neg.sort()

        # cal the target and set the equal one
        index_x, index_y, index_z = np.unravel_index(
            id_pos, (*feature_map_shape, self.anchor_num))
        pos_equal_one[index_x, index_y, index_z] = 1

        # calculate the targets
        targets[index_x, index_y, np.array(index_z) * 7] = \
            (gt_box_center[id_pos_gt, 0] - anchors[id_pos, 0]) / anchors_d[
                id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 1] = \
            (gt_box_center[id_pos_gt, 1] - anchors[id_pos, 1]) / anchors_d[
                id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 2] = \
            (gt_box_center[id_pos_gt, 2] - anchors[id_pos, 2]) / anchors[
                id_pos, 3]
        targets[index_x, index_y, np.array(index_z) * 7 + 3] = np.log(
            gt_box_center[id_pos_gt, 3] / anchors[id_pos, 3])
        targets[index_x, index_y, np.array(index_z) * 7 + 4] = np.log(
            gt_box_center[id_pos_gt, 4] / anchors[id_pos, 4])
        targets[index_x, index_y, np.array(index_z) * 7 + 5] = np.log(
            gt_box_center[id_pos_gt, 5] / anchors[id_pos, 5])
        targets[index_x, index_y, np.array(index_z) * 7 + 6] = (
                gt_box_center[id_pos_gt, 6] - anchors[id_pos, 6])

        index_x, index_y, index_z = np.unravel_index(
            id_neg, (*feature_map_shape, self.anchor_num))
        neg_equal_one[index_x, index_y, index_z] = 1

        # to avoid a box be pos/neg in the same time
        index_x, index_y, index_z = np.unravel_index(
            id_highest, (*feature_map_shape, self.anchor_num))
        neg_equal_one[index_x, index_y, index_z] = 0


        label_dict = {'pos_equal_one': pos_equal_one,
                      'neg_equal_one': neg_equal_one,
                      'targets': targets}

        return label_dict
    
    def generate_label_v2xreal(self, **kwargs):
        """
        Generate targets for training.

        Parameters
        ----------
        argv : list
            gt_box_center:(max_num, 7), anchor:(H, W, anchor_num, 7)

        Returns
        -------
        label_dict : dict
            Dictionary that contains all target related info.
        """
        assert self.params['order'] == 'hwl', 'Currently Voxel only support' \
                                              'hwl bbx order.'
        # (max_num, 8) -- x, y, z, dx, dy, dz, yaw, class
        gt_box_center_all = kwargs['gt_box_center']
        # (H, W, anchor_num, 7)
        anchors_list = kwargs['anchors']

        num_anchors_per_location = kwargs['num_anchors_per_location']
        # (max_num)
        masks = kwargs['mask']

        gt_box_center_all = gt_box_center_all[masks == 1]
        box_cls_labels = []
        box_reg_targets = []
        for i, (anchor_class_name, anchors, anchor_num) in enumerate(zip(self.anchor_class_names, anchors_list, num_anchors_per_location)):
            # Assume the class name order in gt is the same as the one in anchor definition
            gt_box_center = gt_box_center_all[gt_box_center_all[:, -1] -1 == i]
            # (H, W)
            feature_map_shape = anchors.shape[:2]
            # (H*W*anchor_num, 7)
            anchors = anchors.reshape(-1, 7)
            # normalization factor, (H * W * anchor_num)
            anchors_d = np.sqrt(anchors[:, 4] ** 2 + anchors[:, 5] ** 2)

            # (H, W, 2)
            labels = np.ones((*feature_map_shape, anchor_num)) * -1

            pos_equal_one = np.zeros((*feature_map_shape, anchor_num))
            neg_equal_one = np.zeros((*feature_map_shape, anchor_num))
            # # (H, W, self.anchor_num * 7)
            # targets = np.zeros((*feature_map_shape, anchor_num * 7))
            # (n, 8)
            # gt_box_center_valid = gt_box_center[masks == 1]
            gt_box_center_valid = gt_box_center
            # (n, 1)
            gt_box_class_valid = gt_box_center_valid[:, -1:]
            # (n, 8, 3)
            gt_box_corner_valid = \
                box_utils.boxes_to_corners_3d(gt_box_center_valid[:, :7],
                                              self.params['order'])
            # (H*W*anchor_num, 8, 3)
            anchors_corner = \
                box_utils.boxes_to_corners_3d(anchors,
                                              order=self.params['order'])
            # (H*W*anchor_num, 4)
            anchors_standup_2d = \
                box_utils.corner2d_to_standup_box(anchors_corner)
            # (n, 4)
            gt_standup_2d = \
                box_utils.corner2d_to_standup_box(gt_box_corner_valid)

            # (H*W*anchor_n)
            iou = bbox_overlaps(
                np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
                np.ascontiguousarray(gt_standup_2d).astype(np.float32),
            )

            # the anchor boxes has the largest iou across
            # shape: (n)
            id_highest = np.argmax(iou.T, axis=1)
            # [0, 1, 2, ..., n-1]
            id_highest_gt = np.arange(iou.T.shape[0])
            # make sure all highest iou is larger than 0
            mask = iou.T[id_highest_gt, id_highest] > 0
            id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]

            # find anchors iou > params['pos_iou']
            id_pos, id_pos_gt = \
                np.where(iou > self.matched_thresholds[anchor_class_name])
            #  find anchors iou < params['neg_iou']
            id_neg = np.where(np.sum(iou <
                                     self.unmatched_thresholds[anchor_class_name],
                                     axis=1) == iou.shape[1])[0]
            id_pos = np.concatenate([id_pos, id_highest])
            id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])
            id_pos, index = np.unique(id_pos, return_index=True)
            id_pos_gt = id_pos_gt[index]
            id_neg.sort()

            # cal the target and set the equal one
            index_x, index_y, index_z = np.unravel_index(
                id_pos, (*feature_map_shape, anchor_num))
            pos_equal_one[index_x, index_y, index_z] = 1
            labels[index_x, index_y, index_z] = gt_box_center[id_pos_gt, -1]
            # (H, W, anchor_num, 7)
            targets = np.zeros((*feature_map_shape, anchor_num, 7))
            # calculate the targets
            targets[index_x, index_y, np.array(index_z), 0] = \
                (gt_box_center[id_pos_gt, 0] - anchors[id_pos, 0]) / anchors_d[
                    id_pos]
            targets[index_x, index_y, np.array(index_z), 1] = \
                (gt_box_center[id_pos_gt, 1] - anchors[id_pos, 1]) / anchors_d[
                    id_pos]
            targets[index_x, index_y, np.array(index_z), 2] = \
                (gt_box_center[id_pos_gt, 2] - anchors[id_pos, 2]) / anchors[
                    id_pos, 3]
            targets[index_x, index_y, np.array(index_z), 3] = np.log(
                gt_box_center[id_pos_gt, 3] / anchors[id_pos, 3])
            targets[index_x, index_y, np.array(index_z), 4] = np.log(
                gt_box_center[id_pos_gt, 4] / anchors[id_pos, 4])
            targets[index_x, index_y, np.array(index_z), 5] = np.log(
                gt_box_center[id_pos_gt, 5] / anchors[id_pos, 5])
            targets[index_x, index_y, np.array(index_z), 6] = (
                    gt_box_center[id_pos_gt, 6] - anchors[id_pos, 6])


            index_x, index_y, index_z = np.unravel_index(
                id_neg, (*feature_map_shape, anchor_num))
            neg_equal_one[index_x, index_y, index_z] = 1
            labels[index_x, index_y, index_z] = 0



            # to avoid a box be pos/neg in the same time
            index_x, index_y, index_z = np.unravel_index(
                id_highest, (*feature_map_shape, anchor_num))
            neg_equal_one[index_x, index_y, index_z] = 0

            index_x, index_y, index_z = np.unravel_index(
                id_pos, (*feature_map_shape, anchor_num))
            labels[index_x, index_y, index_z] = gt_box_center[id_pos_gt, -1]
            # seems like neg_equal_one is never used throught the optimization

            box_cls_labels.append(labels)
            box_reg_targets.append(targets)




        label_dict = {
            # [(H,W,anchor_num), (H,W,anchor_num)] -> (H, W, anchor_num * num_class)
            'pos_equal_one': np.concatenate(box_cls_labels, axis=-1),
            # [(H,W,anchor_num, 7), (H,W,anchor_num, 7)] -> (H, W, anchor_num * num_class, 7)
            'targets': np.concatenate(box_reg_targets, axis=-2),
            'neg_equal_one': neg_equal_one
        }

        return label_dict

    def generate_pos_region_ranges(self, **kwargs):
        """
        Generate voxel set of targets for training.

        Parameters
        ----------
        argv : dict
            gt_box_center_valid:(max_num, 7)

        Returns
        -------
        label_dict : dict
            Dictionary that contains all target related info.
        """

        assert self.params["order"] == "hwl", (
            "Currently Voxel only support" "hwl bbx order."
        )
        # (n, 7)
        gt_box_center = kwargs["gt_box_center"]

        label_dict = kwargs["label_dict"]
        # (H,W,anchor_num*2)
        targets = label_dict["targets"]

        H, W = targets.shape[:2]
        # (n, 8, 3)
        gt_box_corner_valid = box_utils.boxes_to_corners_3d(
            gt_box_center, self.params["order"]
        )
        # (n,4): (minx,miny,maxx,maxy)
        gt_standup_2d = box_utils.corner2d_to_standup_box(gt_box_corner_valid)

        x = np.linspace(
            self.params["anchor_args"]["cav_lidar_range"][0],
            self.params["anchor_args"]["cav_lidar_range"][3],
            W,
        )
        y = np.linspace(
            self.params["anchor_args"]["cav_lidar_range"][1],
            self.params["anchor_args"]["cav_lidar_range"][4],
            H,
        )

        pos_region_ranges = np.zeros((self.max_num, H, W), dtype=bool)
        for i, item in enumerate(gt_standup_2d):
            left = int(np.argwhere(x <= item[0])[-1])
            right = int(np.argwhere(x >= item[2])[0])
            top = int(np.argwhere(y <= item[1])[-1])
            bottom = int(np.argwhere(y >= item[3])[0])
            pos_region_ranges[[i], top : bottom + 1, left : right + 1] = True

        label_dict.update(
            {
                "pos_region_ranges": pos_region_ranges,
                "cav_num": len(gt_standup_2d),
            }
        )
        return label_dict
    
    def generate_pos_region_rangesv2(self, **kwargs):
        """
        Generate voxel set of targets for training.

        Parameters
        ----------
        argv : dict
            gt_box_center_valid:(max_num, 7)

        Returns
        -------
        label_dict : dict
            Dictionary that contains all target related info.
        """

        assert self.params["order"] == "hwl", (
            "Currently Voxel only support" "hwl bbx order."
        )
        # (n, 4):(minx,miny,maxx,maxy)
        seg_standup_2d= kwargs["gt_box_center"]
        label_dict = kwargs["label_dict"]
        # (H,W,anchor_num*2)
        targets = label_dict["targets"]

        H, W = targets.shape[:2]

        x = np.linspace(
            self.params["anchor_args"]["cav_lidar_range"][0],
            self.params["anchor_args"]["cav_lidar_range"][3],
            W,
        )
        y = np.linspace(
            self.params["anchor_args"]["cav_lidar_range"][1],
            self.params["anchor_args"]["cav_lidar_range"][4],
            H,
        )

        pos_region_ranges = np.zeros((self.max_num, H, W), dtype=bool)
        for i, item in enumerate(seg_standup_2d):
            left = int(np.argwhere(x <= item[0])[-1])
            right = int(np.argwhere(x >= item[2])[0])
            top = int(np.argwhere(y <= item[1])[-1])
            bottom = int(np.argwhere(y >= item[3])[0])
            pos_region_ranges[[i], top : bottom + 1, left : right + 1] = True

        label_dict.update(
            {
                "pos_region_ranges": pos_region_ranges,
                "cav_num": len(seg_standup_2d),
            }
        )
        return label_dict
    @staticmethod
    def collate_batch(label_batch_list):
        """
        Customized collate function for target label generation.

        Parameters
        ----------
        label_batch_list : list
            The list of dictionary  that contains all labels for several
            frames.

        Returns
        -------
        target_batch : dict
            Reformatted labels in torch tensor.
        """
        pos_equal_one = []
        neg_equal_one = []
        # pos_region_ranges = []
        targets = []

        for i in range(len(label_batch_list)):
            pos_equal_one.append(label_batch_list[i]['pos_equal_one'])
            neg_equal_one.append(label_batch_list[i]['neg_equal_one'])
            # pos_region_ranges.append(label_batch_list[i]['pos_region_ranges'])
            targets.append(label_batch_list[i]['targets'])

        pos_equal_one = \
            torch.from_numpy(np.array(pos_equal_one))
        neg_equal_one = \
            torch.from_numpy(np.array(neg_equal_one))
        # pos_region_ranges = \
            # torch.from_numpy(np.array(pos_region_ranges))
        targets = \
            torch.from_numpy(np.array(targets))

        # return {'targets': targets,
        #         'pos_equal_one': pos_equal_one,
        #         'neg_equal_one': neg_equal_one,
        #         'pos_region_ranges': pos_region_ranges}
        return {'targets': targets,
                'pos_equal_one': pos_equal_one,
                'neg_equal_one': neg_equal_one,}
        
    @staticmethod
    def collate_batch_v2xreal(label_batch_list):
        """
        Customized collate function for target label generation.

        Parameters
        ----------
        label_batch_list : list
            The list of dictionary  that contains all labels for several
            frames.

        Returns
        -------
        target_batch : dict
            Reformatted labels in torch tensor.
        """
        pos_equal_one = []
        neg_equal_one = []
        targets = []

        for i in range(len(label_batch_list)):
            pos_equal_one.append(label_batch_list[i]['pos_equal_one'])
            neg_equal_one.append(label_batch_list[i]['neg_equal_one'])
            targets.append(label_batch_list[i]['targets'])

        pos_equal_one = \
            torch.from_numpy(np.array(pos_equal_one))
        neg_equal_one = \
            torch.from_numpy(np.array(neg_equal_one))
        targets = \
            torch.from_numpy(np.array(targets))

        return {'targets': targets,
                'pos_equal_one': pos_equal_one,
                'neg_equal_one': neg_equal_one}

    @staticmethod
    def collate_batch_stamp(label_batch_list):
        """
        Customized collate function for target label generation.

        Parameters
        ----------
        label_batch_list : list
            The list of dictionary  that contains all labels for several
            frames.

        Returns
        -------
        target_batch : dict
            Reformatted labels in torch tensor.
        """
        pos_equal_one = []
        neg_equal_one = []
        targets = []
        for i in range(len(label_batch_list)):
            pos_equal_one.append(torch.tensor(label_batch_list[i]["pos_equal_one"]))
            neg_equal_one.append(torch.tensor(label_batch_list[i]["neg_equal_one"]))
            targets.append(torch.tensor(label_batch_list[i]["targets"]))
        # import pdb; pdb.set_trace()
        # print(pos_equal_one)
        # import pdb; pdb.set_trace()
        if len(pos_equal_one) == 0:
            pos_equal_one = torch.empty(0)
        else:
            pos_equal_one = torch.stack(pos_equal_one, 0)
        if len(pos_equal_one) == 0:
            neg_equal_one = torch.empty(0)
        else:
            neg_equal_one = torch.stack(neg_equal_one, 0)
        if len(pos_equal_one) == 0:
            targets = torch.empty(0)
        else:
            targets = torch.stack(targets, 0)

        return {"targets": targets, "pos_equal_one": pos_equal_one, "neg_equal_one": neg_equal_one}

    @staticmethod
    def collate_batch_pnpda(label_batch_list):
        """
        Customized collate function for target label generation.

        Parameters
        ----------
        label_batch_list : list
            The list of dictionary  that contains all labels for several
            frames.

        Returns
        -------
        target_batch : dict
            Reformatted labels in torch tensor.
        """
        pos_equal_one = []
        neg_equal_one = []
        pos_region_ranges = []
        targets = []

        for i in range(len(label_batch_list)):
            pos_equal_one.append(label_batch_list[i]['pos_equal_one'])
            neg_equal_one.append(label_batch_list[i]['neg_equal_one'])
            pos_region_ranges.append(label_batch_list[i]['pos_region_ranges'])
            targets.append(label_batch_list[i]['targets'])

        pos_equal_one = \
            torch.from_numpy(np.array(pos_equal_one))
        neg_equal_one = \
            torch.from_numpy(np.array(neg_equal_one))
        pos_region_ranges = \
            torch.from_numpy(np.array(pos_region_ranges))
        targets = \
            torch.from_numpy(np.array(targets))

        return {'targets': targets,
                'pos_equal_one': pos_equal_one,
                'neg_equal_one': neg_equal_one,
                'pos_region_ranges': pos_region_ranges}
    
    ## this 3 functions are used for stamp
    def post_process_stamp(self, data_dict, output_dict, agent_idx=0):
        """
        Process the outputs of the model to 2D/3D bounding box.
        Step1: convert each cav's output to bounding box format
        Step2: project the bounding boxes to ego space.
        Step:3 NMS

        For early and intermediate fusion,
            data_dict only contains ego.

        For late fusion,
            data_dcit contains all cavs, so we need transformation matrix.


        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box3d_tensor : torch.Tensor
            The prediction bounding box tensor after NMS.
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor.
        """
        # the final bounding box list
        pred_box3d_list = []
        pred_box2d_list = []
        for cav_id in output_dict.keys():
            if cav_id in data_dict:
                cav_content = data_dict[cav_id]
            else:
                Warning(f"No {cav_id} in data_dict. Using ego instead.")
                cav_content = data_dict["ego"]
            boxes2d_score, projected_boxes3d = self.post_process_single(cav_content, output_dict[cav_id], agent_idx)

        pred_box2d_list.append(boxes2d_score)
        pred_box3d_list.append(projected_boxes3d)

        pred_box3d_tensor, scores = self.post_process_output(pred_box2d_list, pred_box3d_list)

        return pred_box3d_tensor, scores

    def post_process_v2xreal(self, data_dict, output_dict, projection=True):
        """
        Process the outputs of the model to 2D/3D bounding box.
        Step1: convert each cav's output to bounding box format
        Step2: project the bounding boxes to ego space.
        Step:3 NMS

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box3d_tensor : torch.Tensor
            The prediction bounding box tensor after NMS.
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor.
        """
        # the final bounding box list
        pred_box3d_list = []
        pred_box2d_list = []
        pred_label_list = []
        unprojected_box3d_list = []

        for cav_id, cav_content in data_dict.items():
            if cav_id not in output_dict:
                continue
            # the transformation matrix to ego space
            transformation_matrix = cav_content['transformation_matrix']

            # (num_class, H, W, anchor_num, 7)
            all_anchors = cav_content['all_anchors']
            # (H, W, num_class, anchor_num, 7)
            all_anchors = all_anchors.permute(1,2,0,3,4).contiguous()
            # (H*W*num_class*anchor_num, 7)
            all_anchors = all_anchors.view(-1, all_anchors.shape[-1])
            num_anchors = all_anchors.shape[0]

            num_anchors_per_location = cav_content['num_anchors_per_location']

            # classification probability
            # (B, num_anchor*num_class*num_class, H, W)
            prob = output_dict[cav_id]['psm']
            batch_size = prob.shape[0]
            # (B, H, W, num_anchor*num_class*num_class)
            prob = F.sigmoid(prob.permute(0, 2, 3, 1))
            # (B, H*W*num_anchor*num_class, num_class)
            prob = prob.reshape(batch_size, num_anchors, -1)
            # (B, H*W*num_anchor*num_class)
            cls_pred, label_preds = torch.max(prob, dim=-1)
            # class is 1-indexed; 0 is background
            label_preds += 1

            # regression map
            reg = output_dict[cav_id]['rm']
            # (B, H, W, num_anchor*num_class*7)
            reg = reg.permute(0, 2, 3, 1).contiguous()
            # (B, H, W, num_anchor*num_class, 7)
            # reg = reg.reshape(*reg.shape[:3], sum(num_anchors_per_location), -1)
            reg = reg.view(batch_size, num_anchors, -1)

            # convert regression map back to bounding box
            # (N, H*W*num_anchor*num_class, 7)
            batch_box3d = self.delta_to_boxes3d(reg, all_anchors, channel_swap=False)
            mask = \
                torch.gt(cls_pred, self.params['target_args']['score_threshold'])
            mask = mask.view(1, -1)
            mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

            # during validation/testing, the batch size should be 1
            assert batch_box3d.shape[0] == 1
            boxes3d = torch.masked_select(batch_box3d[0],
                                          mask_reg[0]).view(-1, 7)
            # (num_filtered_predicted_box)
            scores = torch.masked_select(cls_pred[0], mask[0])
            # (num_filtered_predicted_box)
            label_preds = torch.masked_select(label_preds[0], mask[0])

            # convert output to bounding box
            if len(boxes3d) != 0:
                # (N, 8, 3)
                boxes3d_corner = \
                    box_utils.boxes_to_corners_3d(boxes3d,
                                                  order=self.params['order'])
                unprojected_box3d_list.append(boxes3d_corner.clone())

                # (N, 8, 3)
                projected_boxes3d = \
                    box_utils.project_box3d(boxes3d_corner,
                                            transformation_matrix)

                # convert 3d bbx to 2d, (N,4)
                projected_boxes2d = \
                    box_utils.corner_to_standup_box_torch(projected_boxes3d)
                # (N, 5)
                boxes2d_score = \
                    torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)

                pred_box2d_list.append(boxes2d_score)
                pred_box3d_list.append(projected_boxes3d)
                pred_label_list.append(label_preds)

        if len(pred_box2d_list) == 0 or len(pred_box3d_list) == 0:
            return None, None
        # shape: (N, 5)
        pred_box2d_list = torch.vstack(pred_box2d_list)
        # scores
        scores = pred_box2d_list[:, -1]
        # predicted 3d bbx
        pred_box3d_tensor = torch.vstack(pred_box3d_list)
        # predicted labels
        pred_label_tensor = torch.cat(pred_label_list)
        # remove large bbx
        keep_index_1 = box_utils.remove_large_pred_bbx(pred_box3d_tensor)
        keep_index_2 = box_utils.remove_bbx_abnormal_z(pred_box3d_tensor)
        keep_index = torch.logical_and(keep_index_1, keep_index_2)
        assert keep_index.sum().cpu() == pred_box3d_tensor.shape[0]

        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        scores = scores[keep_index]
        pred_label_tensor = pred_label_tensor[keep_index]

        unprojected_box3d_tensor = torch.vstack(unprojected_box3d_list)
        unprojected_box3d_tensor = unprojected_box3d_tensor[keep_index]
        # nms
        keep_index = box_utils.nms_rotated(pred_box3d_tensor,
                                           scores,
                                           self.params['nms_thresh']
                                           )
        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        unprojected_box3d_tensor = unprojected_box3d_tensor[keep_index]

        # select cooresponding score
        scores = scores[keep_index]

        # Select corresponding labels
        pred_label_tensor = pred_label_tensor[keep_index]

        # filter out the prediction out of the range.
        mask = \
            box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor)
        pred_box3d_tensor = pred_box3d_tensor[mask, :, :]
        unprojected_box3d_tensor = unprojected_box3d_tensor[mask, :, :]
        scores = scores[mask]
        pred_label_tensor = pred_label_tensor[mask]
        if not projection:
            pred_box3d_tensor = unprojected_box3d_tensor
        assert scores.shape[0] == pred_box3d_tensor.shape[0]

        score_labels = torch.cat([scores.unsqueeze(1), pred_label_tensor.unsqueeze(1)], dim=1)

        return pred_box3d_tensor, score_labels


    def post_process_single(self, cav_content, output_dict, agent_idx=0):
        # the transformation matrix to ego space
        if len(cav_content["transformation_matrix"].shape) > 2:
            transformation_matrix = torch.inverse(cav_content["transformation_matrix"][agent_idx])
        else:
            transformation_matrix = cav_content["transformation_matrix"]

        # rename variable
        if "psm" in output_dict:
            output_dict["cls_preds"] = output_dict["psm"]
        if "rm" in output_dict:
            output_dict["reg_preds"] = output_dict["rm"]
        if "dm" in output_dict:
            output_dict["dir_preds"] = output_dict["dm"]
        # (H, W, anchor_num, 7)
        anchor_box = cav_content["anchor_box"]
        # classification probability
        prob = output_dict["cls_preds"]
        prob = F.sigmoid(prob.permute(0, 2, 3, 1))
        prob = prob.reshape(1, -1)

        # regression map
        reg = output_dict["reg_preds"]
        # convert regression map back to bounding box
        if len(reg.shape) == 4:  # anchor-based. PointPillars, SECOND
            batch_box3d = self.delta_to_boxes3d(reg, anchor_box)
        else:  # anchor-free. CenterPoint
            batch_box3d = reg.view(1, -1, 7)

        mask = torch.gt(prob, self.params["target_args"]["score_threshold"])
        mask = mask.view(1, -1)
        mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

        # during validation/testing, the batch size should be 1
        assert batch_box3d.shape[0] == 1
            
        boxes3d = torch.masked_select(batch_box3d[0], mask_reg[0]).view(-1, 7)
        scores = torch.masked_select(prob[0], mask[0])

        # adding dir classifier
        if "dir_preds" in output_dict.keys() and len(boxes3d) != 0:
            dir_offset = self.params["dir_args"]["dir_offset"]
            num_bins = self.params["dir_args"]["num_bins"]

            dm = output_dict["dir_preds"]  # [N, H, W, 4]
            dir_cls_preds = dm.permute(0, 2, 3, 1).contiguous().reshape(1, -1, num_bins)  # [1, N*H*W*2, 2]
            dir_cls_preds = dir_cls_preds[mask]
            # if rot_gt > 0, then the label is 1, then the regression target is [0, 1]
            dir_labels = torch.max(dir_cls_preds, dim=-1)[
                1
            ]  # indices. shape [1, N*H*W*2].  value 0 or 1. If value is 1, then rot_gt > 0

            period = 2 * np.pi / num_bins  # pi
            dir_rot = limit_period(boxes3d[..., 6] - dir_offset, 0, period)  # 限制在0到pi之间
            boxes3d[..., 6] = dir_rot + dir_offset + period * dir_labels.to(dir_cls_preds.dtype)  # 转化0.25pi到2.5pi
            boxes3d[..., 6] = limit_period(boxes3d[..., 6], 0.5, 2 * np.pi)  # limit to [-pi, pi]

        if "iou_preds" in output_dict.keys() and len(boxes3d) != 0:
            iou = torch.sigmoid(output_dict["iou_preds"].permute(0, 2, 3, 1).contiguous()).reshape(1, -1)
            iou = torch.clamp(iou, min=0.0, max=1.0)
            iou = (iou + 1) * 0.5
            scores = scores * torch.pow(iou.masked_select(mask), 4)

        # convert output to bounding box
        if len(boxes3d) != 0:
            # (N, 8, 3)
            boxes3d_corner = box_utils.boxes_to_corners_3d(boxes3d, order=self.params["order"])

            # STEP 2
            # (N, 8, 3)
            projected_boxes3d = boxes3d_corner
            # projected_boxes3d = box_utils.project_box3d(boxes3d_corner, transformation_matrix)
            # convert 3d bbx to 2d, (N,4)
            projected_boxes2d = box_utils.corner_to_standup_box_torch(projected_boxes3d)
            # (N, 5)
            boxes2d_score = torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)
        else:
            boxes2d_score = torch.zeros((0, 5)).to(boxes3d.device)
            projected_boxes3d = torch.zeros((0, 8, 3)).to(boxes3d.device)

        return boxes2d_score, projected_boxes3d
    def post_process_output(self, pred_box2d_list, pred_box3d_list):
        if len(pred_box2d_list) == 0 or len(pred_box3d_list) == 0:
            return None, None
        # shape: (N, 5)
        pred_box2d_list = torch.vstack(pred_box2d_list)
        # scores
        scores = pred_box2d_list[:, -1]
        # predicted 3d bbx
        pred_box3d_tensor = torch.vstack(pred_box3d_list)
        # remove large bbx
        keep_index_1 = box_utils.remove_large_pred_bbx(pred_box3d_tensor)
        keep_index_2 = box_utils.remove_bbx_abnormal_z(pred_box3d_tensor)
        keep_index = torch.logical_and(keep_index_1, keep_index_2)

        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        scores = scores[keep_index]

        # STEP3
        # nms
        if self.aggregation == "nms":
            keep_index = box_utils.nms_rotated(pred_box3d_tensor, scores, self.params["nms_thresh"])

            pred_box3d_tensor = pred_box3d_tensor[keep_index]

            # select cooresponding score
            scores = scores[keep_index]

            # filter out the prediction out of the range. with z-dim
            pred_box3d_np = pred_box3d_tensor.cpu().numpy()
            pred_box3d_np, mask = box_utils.mask_boxes_outside_range_numpy(
                pred_box3d_np, self.params["gt_range"], order=None, return_mask=True
            )
            pred_box3d_tensor = torch.from_numpy(pred_box3d_np).to(device=pred_box3d_tensor.device)
            scores = scores[mask]

            assert scores.shape[0] == pred_box3d_tensor.shape[0]
        elif self.aggregation == "psa":
            iou_mat = box_utils.compute_self_iou_mat(preds)
            clusters, visited, selected = [], [], []
            for idx, ious in enumerate(iou_mat):
                if idx in visited:
                    continue
                neighbor_idxs = np.nonzero(ious)[0]
                clusters.append(neighbor_idxs)
                visited.extend(neighbor_idxs)
            for cluster in clusters:
                sub_iou_mat = iou_mat[np.ix_(cluster, cluster)]
                sub_probs = probs[cluster]
                values = sub_iou_mat.dot(sub_probs)
                soft_bools = softmax(values / 1e-6)
                bools = soft_bools > 0.5
                selected.extend(cluster[bools])
            preds = preds[selected]
            probs = probs[selected]

        return pred_box3d_tensor, scores
    
    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.
        Step1: convert each cav's output to bounding box format
        Step2: project the bounding boxes to ego space.
        Step:3 NMS

        For early and intermediate fusion,
            data_dict only contains ego.

        For late fusion,
            data_dcit contains all cavs, so we need transformation matrix.


        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box3d_tensor : torch.Tensor
            The prediction bounding box tensor after NMS.
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor.
        """
        # the final bounding box list
        pred_box3d_list = []
        pred_box2d_list = []
        for cav_id in output_dict.keys():
            assert cav_id in data_dict
            cav_content = data_dict[cav_id]
            # the transformation matrix to ego space
            transformation_matrix = cav_content['transformation_matrix'] # no clean

            # rename variable 
            if 'psm' in output_dict[cav_id]:
                output_dict[cav_id]['cls_preds'] = output_dict[cav_id]['psm']
            if 'rm' in output_dict:
                output_dict[cav_id]['reg_preds'] = output_dict[cav_id]['rm']
            if 'dm' in output_dict:
                output_dict[cav_id]['dir_preds'] = output_dict[cav_id]['dm']

            # (H, W, anchor_num, 7)
            anchor_box = cav_content['anchor_box']

            # classification probability
            prob = output_dict[cav_id]['cls_preds']
            prob = F.sigmoid(prob.permute(0, 2, 3, 1))
            prob = prob.reshape(1, -1)

            # regression map
            reg = output_dict[cav_id]['reg_preds']

            # convert regression map back to bounding box
            if len(reg.shape) == 4: # anchor-based. PointPillars, SECOND
                batch_box3d = self.delta_to_boxes3d(reg, anchor_box)
            else: # anchor-free. CenterPoint
                batch_box3d = reg.view(1, -1, 7)

            mask = \
                torch.gt(prob, self.params['target_args']['score_threshold'])
            mask = mask.view(1, -1)
            mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

            # during validation/testing, the batch size should be 1
            assert batch_box3d.shape[0] == 1
            boxes3d = torch.masked_select(batch_box3d[0],
                                          mask_reg[0]).view(-1, 7)
            scores = torch.masked_select(prob[0], mask[0])

            # adding dir classifier
            if 'dir_preds' in output_dict[cav_id].keys() and len(boxes3d) !=0:
                dir_offset = self.params['dir_args']['dir_offset']
                num_bins = self.params['dir_args']['num_bins']


                dm  = output_dict[cav_id]['dir_preds'] # [N, H, W, 4]
                dir_cls_preds = dm.permute(0, 2, 3, 1).contiguous().reshape(1, -1, num_bins) # [1, N*H*W*2, 2]
                dir_cls_preds = dir_cls_preds[mask]
                # if rot_gt > 0, then the label is 1, then the regression target is [0, 1]
                dir_labels = torch.max(dir_cls_preds, dim=-1)[1]  # indices. shape [1, N*H*W*2].  value 0 or 1. If value is 1, then rot_gt > 0
                
                period = (2 * np.pi / num_bins) # pi
                dir_rot = limit_period(
                    boxes3d[..., 6] - dir_offset, 0, period
                ) # 限制在0到pi之间
                boxes3d[..., 6] = dir_rot + dir_offset + period * dir_labels.to(dir_cls_preds.dtype) # 转化0.25pi到2.5pi
                boxes3d[..., 6] = limit_period(boxes3d[..., 6], 0.5, 2 * np.pi) # limit to [-pi, pi]

            if 'iou_preds' in output_dict[cav_id].keys() and len(boxes3d) != 0:
                iou = torch.sigmoid(output_dict[cav_id]['iou_preds'].permute(0, 2, 3, 1).contiguous()).reshape(1, -1)
                iou = torch.clamp(iou, min=0.0, max=1.0)
                iou = (iou + 1) * 0.5
                scores = scores * torch.pow(iou.masked_select(mask), 4)

            # convert output to bounding box
            if len(boxes3d) != 0:
                # (N, 8, 3)
                boxes3d_corner = \
                    box_utils.boxes_to_corners_3d(boxes3d,
                                                  order=self.params['order'])
                
                # STEP 2
                # (N, 8, 3)
                projected_boxes3d = \
                    box_utils.project_box3d(boxes3d_corner,
                                            transformation_matrix)
                # convert 3d bbx to 2d, (N,4)
                projected_boxes2d = \
                    box_utils.corner_to_standup_box_torch(projected_boxes3d)
                # (N, 5)
                boxes2d_score = \
                    torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)

                pred_box2d_list.append(boxes2d_score)
                pred_box3d_list.append(projected_boxes3d)

        if len(pred_box2d_list) ==0 or len(pred_box3d_list) == 0:
            return None, None
        # shape: (N, 5)
        pred_box2d_list = torch.vstack(pred_box2d_list)
        # scores
        scores = pred_box2d_list[:, -1]
        # predicted 3d bbx
        pred_box3d_tensor = torch.vstack(pred_box3d_list)
        # remove large bbx
        keep_index_1 = box_utils.remove_large_pred_bbx(pred_box3d_tensor)
        keep_index_2 = box_utils.remove_bbx_abnormal_z(pred_box3d_tensor)
        keep_index = torch.logical_and(keep_index_1, keep_index_2)

        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        scores = scores[keep_index]

        # STEP3
        # nms
        keep_index = box_utils.nms_rotated(pred_box3d_tensor,
                                           scores,
                                           self.params['nms_thresh']
                                           )

        pred_box3d_tensor = pred_box3d_tensor[keep_index]

        # select cooresponding score
        scores = scores[keep_index]
        
        # filter out the prediction out of the range. with z-dim
        pred_box3d_np = pred_box3d_tensor.cpu().numpy()
        pred_box3d_np, mask = box_utils.mask_boxes_outside_range_numpy(pred_box3d_np,
                                                    self.params['gt_range'],
                                                    order=None,
                                                    return_mask=True)
        pred_box3d_tensor = torch.from_numpy(pred_box3d_np).to(device=pred_box3d_tensor.device)
        scores = scores[mask]

        assert scores.shape[0] == pred_box3d_tensor.shape[0]

        return pred_box3d_tensor, scores

    
    def post_process_online(self, data_dict):
        ## This is used for v2xreal
        
        """
        Process the outputs of the model to 2D/3D bounding box.
        Step1: convert each cav's output to bounding box format
        Step2: project the bounding boxes to ego space.
        Step:3 NMS

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box3d_tensor : torch.Tensor
            The prediction bounding box tensor after NMS.
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor.
        """
        # the final bounding box list
        pred_box3d_list = []
        pred_box2d_list = []

        for cav_id, cav_content in data_dict.items():
            # the transformation matrix to ego space
            transformation_matrix = cav_content['transformation_matrix']

            prob = cav_content["score"].reshape(1, -1)
            # (N, W*L*anchor_num, 7)
            batch_box3d = cav_content["box"]
            mask = \
                torch.gt(prob, self.params['target_args']['score_threshold'])
            mask = mask.view(1, -1)
            mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

            # during validation/testing, the batch size should be 1
            assert batch_box3d.shape[0] == 1
            boxes3d = torch.masked_select(batch_box3d[0],
                                          mask_reg[0]).view(-1, 7)
            scores = torch.masked_select(prob[0], mask[0])

            # convert output to bounding box
            if len(boxes3d) != 0:
                # (N, 8, 3)
                boxes3d_corner = \
                    box_utils.boxes_to_corners_3d(boxes3d,
                                                  order=self.params['order'])
                # (N, 8, 3)
                projected_boxes3d = \
                    box_utils.project_box3d(boxes3d_corner,
                                            transformation_matrix)
                # convert 3d bbx to 2d, (N,4)
                projected_boxes2d = \
                    box_utils.corner_to_standup_box_torch(projected_boxes3d)
                # (N, 5)
                boxes2d_score = \
                    torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)

                pred_box2d_list.append(boxes2d_score)
                pred_box3d_list.append(projected_boxes3d)

        if len(pred_box2d_list) == 0 or len(pred_box3d_list) == 0:
            return None, None
        # shape: (N, 5)
        pred_box2d_list = torch.vstack(pred_box2d_list)
        # scores
        scores = pred_box2d_list[:, -1]
        # predicted 3d bbx
        pred_box3d_tensor = torch.vstack(pred_box3d_list)
        # remove large bbx
        keep_index_1 = box_utils.remove_large_pred_bbx(pred_box3d_tensor)
        keep_index_2 = box_utils.remove_bbx_abnormal_z(pred_box3d_tensor)
        keep_index = torch.logical_and(keep_index_1, keep_index_2)

        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        scores = scores[keep_index]

        # nms
        keep_index = box_utils.nms_rotated(pred_box3d_tensor,
                                           scores,
                                           self.params['nms_thresh']
                                           )

        pred_box3d_tensor = pred_box3d_tensor[keep_index]

        # select cooresponding score
        scores = scores[keep_index]

        # filter out the prediction out of the range.
        mask = \
            box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor)
        pred_box3d_tensor = pred_box3d_tensor[mask, :, :]
        scores = scores[mask]

        assert scores.shape[0] == pred_box3d_tensor.shape[0]

        return pred_box3d_tensor, scores

    @staticmethod
    def delta_to_boxes3d(deltas, anchors):
        """
        Convert the output delta to 3d bbx.

        Parameters
        ----------
        deltas : torch.Tensor
            (N, 14, H, W)
        anchors : torch.Tensor
            (W, L, 2, 7) -> xyzhwlr

        Returns
        -------
        box3d : torch.Tensor
            (N, W*L*2, 7)
        """
        # batch size
        N = deltas.shape[0]
        deltas = deltas.permute(0, 2, 3, 1).contiguous().view(N, -1, 7)
        boxes3d = torch.zeros_like(deltas)

        if deltas.is_cuda:
            anchors = anchors.cuda()
            boxes3d = boxes3d.cuda()

        # (W*L*2, 7)
        anchors_reshaped = anchors.view(-1, 7).float()
        # the diagonal of the anchor 2d box, (W*L*2)
        anchors_d = torch.sqrt(
            anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2)
        anchors_d = anchors_d.repeat(N, 2, 1).transpose(1, 2)
        anchors_reshaped = anchors_reshaped.repeat(N, 1, 1)

        # Inv-normalize to get xyz
        boxes3d[..., [0, 1]] = torch.mul(deltas[..., [0, 1]], anchors_d) + \
                               anchors_reshaped[..., [0, 1]]
        boxes3d[..., [2]] = torch.mul(deltas[..., [2]],
                                      anchors_reshaped[..., [3]]) + \
                            anchors_reshaped[..., [2]]
        # hwl
        boxes3d[..., [3, 4, 5]] = torch.exp(
            deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]
        # yaw angle
        boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

        return boxes3d

    @staticmethod
    def delta_to_boxes3d_v2xreal(deltas, anchors, channel_swap=True):
        """
        Convert the output delta to 3d bbx.

        Parameters
        ----------
        deltas : torch.Tensor
            (N, W, L, 14)
        anchors : torch.Tensor
            (W, L, 2, 7) -> xyzhwlr
        channel_swap : bool
            Whether to swap the channel of deltas. It is only false when using
            FPV-RCNN

        Returns
        -------
        box3d : torch.Tensor
            (N, W*L*2, 7)
        """
        # batch size
        N = deltas.shape[0]
        if channel_swap:
            deltas = deltas.permute(0, 2, 3, 1).contiguous().view(N, -1, 7)
        else:
            # (B, W*L*2, 7)
            deltas = deltas.contiguous().view(N, -1, 7)

        boxes3d = torch.zeros_like(deltas)
        if deltas.is_cuda:
            anchors = anchors.cuda()
            boxes3d = boxes3d.cuda()

        # (W*L*2, 7)
        anchors_reshaped = anchors.view(-1, 7).float()
        # the diagonal of the anchor 2d box, (W*L*2)
        anchors_d = torch.sqrt(
            anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2)
        # (B, W*L*2, 2)
        anchors_d = anchors_d.repeat(N, 2, 1).transpose(1, 2)
        # (B, W*L*2, 7)
        anchors_reshaped = anchors_reshaped.repeat(N, 1, 1)

        # Inv-normalize to get xyz
        boxes3d[..., [0, 1]] = torch.mul(deltas[..., [0, 1]], anchors_d) + \
                               anchors_reshaped[..., [0, 1]]
        boxes3d[..., [2]] = torch.mul(deltas[..., [2]],
                                      anchors_reshaped[..., [3]]) + \
                            anchors_reshaped[..., [2]]
        # hwl
        boxes3d[..., [3, 4, 5]] = torch.exp(
            deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]
        # yaw angle
        boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

        return boxes3d

    @staticmethod
    def visualize(pred_box_tensor, gt_tensor, pcd, show_vis, save_path, dataset=None):
        """
        Visualize the prediction, ground truth with point cloud together.

        Parameters
        ----------
        pred_box_tensor : torch.Tensor
            (N, 8, 3) prediction.

        gt_tensor : torch.Tensor
            (N, 8, 3) groundtruth bbx

        pcd : torch.Tensor
            PointCloud, (N, 4).

        show_vis : bool
            Whether to show visualization.

        save_path : str
            Save the visualization results to given path.

        dataset : BaseDataset
            opencood dataset object.

        """
        vis_utils.visualize_single_sample_output_gt(pred_box_tensor,
                                                    gt_tensor,
                                                    pcd,
                                                    show_vis,
                                                    save_path)
