# -*- coding: utf-8 -*-
# Author: Yifan Lu
# Add direction classification loss
# The originally point_pillar_loss.py, can not determine if the box heading is opposite to the GT.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from opencood.utils.common_utils import limit_period
from opencood.loss.point_pillar_depth_loss import PointPillarDepthLoss
from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from icecream import ic
import wandb

class PointPillarDiffcommLoss(PointPillarDepthLoss):
    def __init__(self, args):
        super(PointPillarDiffcommLoss, self).__init__(args)

        self.generate_weight = args['generate_weight']


    def forward(self, output_dict, target_dict, suffix=""):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        if 'record_len' in output_dict:
            batch_size = int(output_dict['record_len'].sum())
        elif 'batch_size' in output_dict:
            batch_size = output_dict['batch_size']
        else:
            batch_size = target_dict['pos_equal_one'].shape[0]
        
        # pos_equal_one: [B, H, W, 2] neg_equal_one: [B, H, W, 2]
        ## jugde the positive and negative 
        # pos_is_binary = torch.all(torch.logical_or(target_dict['pos_equal_one'] == 0, target_dict['pos_equal_one'] == 1))
        # neg_is_binary = torch.all(torch.logical_or(target_dict['neg_equal_one'] == 0, target_dict['neg_equal_one'] == 1))
        # print(torch.equal(target_dict['pos_equal_one'], 1 - target_dict['neg_equal_one']))
        # diff_count = torch.sum(target_dict['pos_equal_one'].ne(1 - target_dict['neg_equal_one']))

        total_loss = super().forward(output_dict, target_dict, suffix)

        gt_feature = output_dict['gt_feature']
        pred_feature = output_dict['pred_feature']
        diff_loss = nn.MSELoss()
        generate_loss = diff_loss(gt_feature, pred_feature)
        self.loss_dict.update({'generate_loss': generate_loss.item()})

        total_loss = total_loss + self.generate_weight * generate_loss

        self.loss_dict.update({'total_loss': total_loss.item(),
                               'generate_loss': generate_loss.item()})

        return total_loss


    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * \
                            torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * \
                          torch.sin(boxes2[..., dim:dim + 1])

        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding,
                            boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding,
                            boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    def get_direction_target(self, reg_targets):
        """
        Args:
            reg_targets:  [N, H * W * #anchor_num, 7]
                The last term is (theta_gt - theta_a)
        
        Returns:
            dir_targets:
                theta_gt: [N, H * W * #anchor_num, NUM_BIN] 
                NUM_BIN = 2
        """
        num_bins = self.dir['args']['num_bins']
        dir_offset = self.dir['args']['dir_offset']
        anchor_yaw = np.deg2rad(np.array(self.dir['args']['anchor_yaw']))  # for direction classification
        self.anchor_yaw_map = torch.from_numpy(anchor_yaw).view(1,-1,1)  # [1,2,1]
        self.anchor_num = self.anchor_yaw_map.shape[1]

        H_times_W_times_anchor_num = reg_targets.shape[1]
        anchor_map = self.anchor_yaw_map.repeat(1, H_times_W_times_anchor_num//self.anchor_num, 1).to(reg_targets.device) # [1, H * W * #anchor_num, 1]
        rot_gt = reg_targets[..., -1] + anchor_map[..., -1] # [N, H*W*anchornum]
        offset_rot = limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()  # [N, H*W*anchornum]
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
        # one_hot:
        # if rot_gt > 0, then the label is 1, then the regression target is [0, 1]
        dir_cls_targets = one_hot_f(dir_cls_targets, num_bins)
        return dir_cls_targets



    def logging(self, epoch, batch_id, batch_len, writer = None, suffix="", iter=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict.get('total_loss', 0)
        reg_loss = self.loss_dict.get('reg_loss', 0)
        cls_loss = self.loss_dict.get('cls_loss', 0)
        dir_loss = self.loss_dict.get('dir_loss', 0)
        iou_loss = self.loss_dict.get('iou_loss', 0)
        depth_loss = self.loss_dict.get('depth_loss', 0)
        gen_loss = self.loss_dict.get('generate_loss', 0)


        print("[epoch %d][%d/%d]%s || Loss: %.4f || Conf Loss: %.4f"
              " || Loc Loss: %.4f || Dir Loss: %.4f || IoU Loss: %.4f || Depth Loss: %.4f || Gen Loss: %.4f" % (
                  epoch, batch_id + 1, batch_len, suffix,
                  total_loss, cls_loss, reg_loss, dir_loss, iou_loss, depth_loss, gen_loss))

        if not writer is None:
            writer.add_scalar('Regression_loss'+suffix, reg_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Confidence_loss'+suffix, cls_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Dir_loss'+suffix, dir_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Iou_loss'+suffix, iou_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Gen_loss'+suffix, iou_loss,
                            epoch*batch_len + batch_id)
            
        wandb.log({ 'train_loss_iter'+suffix: total_loss,
                    'Reg_loss'+suffix: reg_loss,
                    'Conf_loss'+suffix: cls_loss,
                    'Dir_loss'+suffix: dir_loss,
                    'Iou_loss'+suffix: iou_loss,
                    'depth_loss'+suffix: depth_loss,
                    'Gen_loss'+suffix: gen_loss}, step=iter)

def one_hot_f(tensor, num_bins, dim=-1, on_value=1.0, dtype=torch.float32):
    tensor_onehot = torch.zeros(*list(tensor.shape), num_bins, dtype=dtype, device=tensor.device) 
    tensor_onehot.scatter_(dim, tensor.unsqueeze(dim).long(), on_value)                    
    return tensor_onehot

def softmax_cross_entropy_with_logits(logits, labels):
    param = list(range(len(logits.shape)))
    transpose_param = [0] + [param[-1]] + param[1:-1]
    logits = logits.permute(*transpose_param)
    loss_ftor = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_ftor(logits, labels.max(dim=-1)[1])
    return loss

def weighted_smooth_l1_loss(preds, targets, sigma=3.0, weights=None):
    diff = preds - targets
    abs_diff = torch.abs(diff)
    abs_diff_lt_1 = torch.le(abs_diff, 1 / (sigma ** 2)).type_as(abs_diff)
    loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * sigma, 2) + \
               (abs_diff - 0.5 / (sigma ** 2)) * (1.0 - abs_diff_lt_1)
    if weights is not None:
        loss *= weights
    return loss
    ## formula: 0.5 * (|x|*sigma)^2 if |x| < 1/sigma^2 else |x| - 0.5/sigma^2, where x = preds - targets.



def sigmoid_focal_loss(preds, targets, weights=None, **kwargs):
    assert 'gamma' in kwargs and 'alpha' in kwargs
    # sigmoid cross entropy with logits
    # more details: https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    per_entry_cross_ent = torch.clamp(preds, min=0) - preds * targets.type_as(preds)
    per_entry_cross_ent += torch.log1p(torch.exp(-torch.abs(preds)))
    # focal loss
    prediction_probabilities = torch.sigmoid(preds)
    p_t = (targets * prediction_probabilities) + ((1 - targets) * (1 - prediction_probabilities))
    modulating_factor = torch.pow(1.0 - p_t, kwargs['gamma'])
    alpha_weight_factor = targets * kwargs['alpha'] + (1 - targets) * (1 - kwargs['alpha'])

    loss = modulating_factor * alpha_weight_factor * per_entry_cross_ent
    if weights is not None:
        loss *= weights
    return loss