""" Author: Yifan Lu <yifan_lu@sjtu.edu.cn>

HEAL: An Extensible Framework for Open Heterogeneous Collaborative Perception 
"""

import torch
import torch.nn as nn
import numpy as np
from icecream import ic
from collections import OrderedDict, Counter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.models.diffcomm_modules.cond_diff import DiffComm
from opencood.models.diffcomm_modules.enhance_v12 import Enhancerv12
from opencood.models.diffcomm_modules.message_extractor_v2 import MessageExtractorv2
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
from opencood.visualization.vis_bevfeat import vis_bev, visualize_feature_maps
import importlib
import torchvision
from opencood.visualization.vis_bevfeat import vis_bev
from opencood.models.fuse_modules.fusion_in_one import regroup

class HeterPyramidCollabWDiffComm(nn.Module):
    def __init__(self, args):
        super(HeterPyramidCollabWDiffComm, self).__init__()
        self.args = args
        self.diffcomm = DiffComm(args['diffcomm'])
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] 
        self.modality_name_list = modality_name_list
        self.ego_modality = args['ego_modality']
        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()

        self.cam_crop_info = {} 

        # setup each modality model
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name

            # import model
            encoder_filename = "opencood.models.heter_encoders"
            encoder_lib = importlib.import_module(encoder_filename)
            encoder_class = None
            target_model_name = model_setting['core_method'].replace('_', '')

            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls

            """
            Encoder building
            """
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            if model_setting['encoder_args'].get("depth_supervision", False):
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)

            """
            Backbone building 
            """
            setattr(self, f"backbone_{modality_name}", ResNetBEVBackbone(model_setting['backbone_args']))
            setattr(self, f"message_extractor_{modality_name}", MessageExtractorv2(64, 2))

            """
            Aligner building
            """
            setattr(self, f"aligner_{modality_name}", AlignNet(model_setting['aligner_args']))
            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}", (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}", (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))
                setattr(self, f"xdist_{modality_name}", (camera_mask_args['grid_conf']['xbound'][1] - camera_mask_args['grid_conf']['xbound'][0]))
                setattr(self, f"ydist_{modality_name}", (camera_mask_args['grid_conf']['ybound'][1] - camera_mask_args['grid_conf']['ybound'][0]))
                self.cam_crop_info[modality_name] = {
                    f"crop_ratio_W_{modality_name}": eval(f"self.crop_ratio_W_{modality_name}"),
                    f"crop_ratio_H_{modality_name}": eval(f"self.crop_ratio_H_{modality_name}"),
                }

        """For feature transformation"""
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1

        self.gmatch = False
        if 'gmatch' in args and args['gmatch']:
            self.gmatch = True
        
        """
        Fusion, by default multiscale fusion: 
        Note the input of PyramidFusion has downsampled 2x. (SECOND required)
        """
        self.pyramid_backbone = PyramidFusion(args['fusion_backbone'])


        """
        Shrink header
        """
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        """
        Shared Heads
        """
        self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'],
                                  kernel_size=1)
        self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
        if 'enhancer' in args and args['enhancer'] == 'enhancev12':
            self.enhancer = Enhancerv12(64, [8, 8], 4)
            print("use enhancev12")        

        # compressor will be only trainable
        self.compress = False
        if 'compressor' in args:
            self.compress = True
            self.compressor = NaiveCompressor(args['compressor']['input_dim'],
                                              args['compressor']['compress_ratio'])

        self.model_train_init()
        # check again which module is not fixed.
        check_trainable_module(self)


    def model_train_init(self):
        # if compress, only make compressor trainable
        if self.compress:
            # freeze all
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
            # unfreeze compressor
            self.compressor.train()
            for p in self.compressor.parameters():
                p.requires_grad_(True)

    def forward(self, data_dict):
        output_dict = {'pyramid': 'collab'}
        agent_modality_list = data_dict['agent_modality_list'] 
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len'] 
        # print(agent_modality_list)
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}
        modality_message_dict = {}

        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
            feature = eval(f"self.aligner_{modality_name}")(feature)
            message = eval(f"self.message_extractor_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature
            modality_message_dict[modality_name] = message

        """
        Crop/Padd camera feature map.
        """
        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict:
                if self.sensor_type_dict[modality_name] == "camera":
                    # should be padding. Instead of masking
                    feature = modality_feature_dict[modality_name]
                    _, _, H, W = feature.shape
                    target_H = int(H*eval(f"self.crop_ratio_H_{modality_name}"))
                    target_W = int(W*eval(f"self.crop_ratio_W_{modality_name}"))

                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    modality_feature_dict[modality_name] = crop_func(feature)
                    if eval(f"self.depth_supervision_{modality_name}"):
                        output_dict.update({
                            f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items
                        })

        """
        Assemble heter features
        """
        counting_dict = {modality_name:0 for modality_name in self.modality_name_list}
        heter_feature_2d_list = []
        heter_message_list = []
        for modality_name in agent_modality_list:
            feat_idx = counting_dict[modality_name]
            heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])
            heter_message_list.append(modality_message_dict[modality_name][feat_idx])
            counting_dict[modality_name] += 1

        heter_feature_2d = torch.stack(heter_feature_2d_list)
        heter_message = torch.stack(heter_message_list)
        
        if self.compress:
            heter_feature_2d = self.compressor(heter_feature_2d)

        # heter_feature_2d is downsampled 2x
        # add croping information to collaboration module
        # note='m1m3_pyramid_'
        # vis_bev(heter_feature_2d[0].detach().cpu().numpy(), type=note + 'ego', normalize=True)
        # vis_bev(heter_feature_2d[1].detach().cpu().numpy(), type=note + 'cav', normalize=True)
        # visualize_feature_maps(heter_feature_2d, mode='mean', save_path=f'/home/junfei.zhou/DATACENTER2/data/code/DiffComm/feat_vis_m1m3/{note}ego.png')
        
        conditions = heter_message
        """
        Single supervision
        """
        if self.supervise_single:
            cls_preds_before_fusion = self.cls_head_single(heter_feature_2d)
            reg_preds_before_fusion = self.reg_head_single(heter_feature_2d)
            dir_preds_before_fusion = self.dir_head_single(heter_feature_2d)
            output_dict.update({'cls_preds_single': cls_preds_before_fusion,
                                'reg_preds_single': reg_preds_before_fusion,
                                'dir_preds_single': dir_preds_before_fusion})

        
        spatial_mask = torch.any(heter_feature_2d, dim=1).to(torch.uint8).unsqueeze(1).to(heter_feature_2d.device)
        gt_feature = heter_feature_2d

        gen_data_dict = self.diffcomm(heter_feature_2d, conditions, record_len)
        pred_feature = gen_data_dict['pred_feature'] * spatial_mask
        output_dict.update({'gt_feature': gt_feature,
                            'pred_feature': pred_feature})
        
        # note = 'm3_diffcomm'
        # vis_bev(conditions[0].squeeze(0).detach().cpu().numpy(), type=note + 'ego_condi')
        # vis_bev(conditions[1].squeeze(0).detach().cpu().numpy(), type=note + 'cav_condi')
        # vis_bev(gt_feature[0].detach().cpu().numpy(), type=note + 'ego')
        # vis_bev(gen_data_dict['t1'].squeeze(0).detach().cpu().numpy(), type=note + 'ego_noisy_t1')
        # vis_bev(gen_data_dict['t2'].squeeze(0).detach().cpu().numpy(), type=note + 'ego_noisy_t2')
        # vis_bev(gt_feature[1].detach().cpu().numpy(), type=note + 'cav')
        # vis_bev(gen_data_dict['pred_feature'][0].detach().cpu().numpy(), type=note + 'ego_gen')
        # vis_bev(gen_data_dict['pred_feature'][1].detach().cpu().numpy(), type=note + 'cav_gen')
        
        heter_feature_2d = pred_feature
        
        ###replace ego feat ure with gt_feature
        split_gt_feature = regroup(gt_feature, record_len)
        split_pred_feature = regroup(heter_feature_2d, record_len)
        ego_index = 0
        for index in range(len(split_gt_feature)):
            heter_feature_2d[ego_index] = split_gt_feature[index][0]
            ego_index = ego_index + split_gt_feature[index].shape[0]
    
        
 
        if len(heter_feature_2d.shape) == 3:
            heter_feature_2d = heter_feature_2d.unsqueeze(0) ## for the case of bs=1 and only ego
        if hasattr(self, 'enhancer'):
            heter_feature_2d = self.enhancer(heter_feature_2d, affine_matrix, record_len)
            if self.gmatch:
                gt_feature = self.enhancer(gt_feature, affine_matrix, record_len)
        
        
        fused_feature, occ_outputs = self.pyramid_backbone.forward_collab(
                                                heter_feature_2d,
                                                record_len, 
                                                affine_matrix, 
                                                agent_modality_list, 
                                                self.cam_crop_info
                                            )
        if self.gmatch:
            fused_feature_T = self.pyramid_backbone.forward_collab(
                                                gt_feature,
                                                record_len, 
                                                affine_matrix, 
                                                agent_modality_list, 
                                                self.cam_crop_info
                                            )

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)
            if self.gmatch:
                fused_feature_T = self.shrink_conv(fused_feature_T)

        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        dir_preds = self.dir_head(fused_feature)
        if self.gmatch:
            cls_preds_T = self.cls_head(fused_feature_T)
            reg_preds_T = self.reg_head(fused_feature_T)
            dir_preds_T = self.dir_head(fused_feature_T)

            output_dict.update({'cls_preds_T': cls_preds_T,
                                'reg_preds_T': reg_preds_T,
                                'dir_preds_T': dir_preds_T})

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})
        
        output_dict.update({'occ_single_list': 
                            occ_outputs})

        return output_dict
