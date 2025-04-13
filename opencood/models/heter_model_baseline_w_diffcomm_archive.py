# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

# A unified framework for LiDAR-only / Camera-only / Heterogeneous collaboration.
# Support multiple fusion strategies.


import torch
import torch.nn as nn
import numpy as np
from icecream import ic
from collections import OrderedDict, Counter
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.fusion_in_one import MaxFusion, AttFusion, DiscoFusion, V2VNetFusion, V2XViTFusion, CoBEVT, Where2commFusion, Who2comFusion
from opencood.models.diffcomm_modules.cond_diff import DiffComm
# from opencood.models.diffcomm_modules.message_extractor import MessageExtractor
from opencood.models.diffcomm_modules.message_extractor import MessageExtractor
from opencood.models.diffcomm_modules.message_extractor_v2 import MessageExtractorv2
from opencood.models.diffcomm_modules.enhance import Enhancer
from opencood.models.diffcomm_modules.enhance_v1 import Enhancerv1
from opencood.models.diffcomm_modules.enhance_v2 import Enhancerv2
from opencood.models.diffcomm_modules.enhance_v3 import Enhancerv3
from opencood.models.diffcomm_modules.enhance_v4 import Enhancerv4
from opencood.models.diffcomm_modules.enhance_v5 import Enhancerv5
from opencood.models.diffcomm_modules.enhance_v6 import Enhancerv6
from opencood.models.diffcomm_modules.enhance_v7 import Enhancerv7
from opencood.models.diffcomm_modules.enhance_v8 import Enhancerv8
from opencood.models.diffcomm_modules.enhance_v9 import Enhancerv9
from opencood.models.diffcomm_modules.enhance_v10 import Enhancerv10
from opencood.models.diffcomm_modules.enhance_v11 import Enhancerv11
from opencood.models.diffcomm_modules.enhance_v12 import Enhancerv12
from opencood.models.diffcomm_modules.enhance_v13 import Enhancerv13

from opencood.utils.diffcomm_utils import get_distance_bias, get_angle_bias
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.models.fuse_modules.fusion_in_one import regroup
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
import importlib
import torchvision
from opencood.visualization.simple_vis import visualize
from opencood.visualization.vis_bevfeat import vis_bev

class HeterModelBaselineWDiffCommArchive(nn.Module):
    def __init__(self, args):
        super(HeterModelBaselineWDiffCommArchive, self).__init__()
        self.args = args
        
        self.diffcomm = DiffComm(args['diffcomm'])
        
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] 
        self.modality_name_list = modality_name_list

        self.ego_modality = args['ego_modality']

        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()

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
            setattr(self, f"backbone_{modality_name}", BaseBEVBackbone(model_setting['backbone_args'], 
                                                                       model_setting['backbone_args'].get('inplanes',64)))

            """
            shrink conv building
            """
            setattr(self, f"shrinker_{modality_name}", DownsampleConv(model_setting['shrink_header']))

            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}", (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}", (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))
                setattr(self, f"xdist_{modality_name}", (camera_mask_args['grid_conf']['xbound'][1] - camera_mask_args['grid_conf']['xbound'][0]))
                setattr(self, f"ydist_{modality_name}", (camera_mask_args['grid_conf']['ybound'][1] - camera_mask_args['grid_conf']['ybound'][0]))

        """For feature transformation"""
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1
        
        self.gmatch = False
        if 'gmatch' in args and args['gmatch']:
            self.gmatch = True

        self.supervise_single = False
        if args.get("supervise_single", False):
            self.supervise_single = True
            in_head_single = args['in_head_single']
            setattr(self, f'cls_head_single', nn.Conv2d(in_head_single, args['anchor_number'], kernel_size=1))
            setattr(self, f'reg_head_single', nn.Conv2d(in_head_single, args['anchor_number'] * 7, kernel_size=1))
            setattr(self, f'dir_head_single', nn.Conv2d(in_head_single, args['anchor_number'] *  args['dir_args']['num_bins'], kernel_size=1))


        if args['fusion_method'] == "max":
            self.fusion_net = MaxFusion()
        if args['fusion_method'] == "att":
            self.fusion_net = AttFusion(args['att']['feat_dim'])
        if args['fusion_method'] == "disconet":
            self.fusion_net = DiscoFusion(args['disconet']['feat_dim'])
        if args['fusion_method'] == "v2vnet":
            self.fusion_net = V2VNetFusion(args['v2vnet'])
        if args['fusion_method'] == 'v2xvit':
            self.fusion_net = V2XViTFusion(args['v2xvit'])
        if args['fusion_method'] == 'cobevt':
            self.fusion_net = CoBEVT(args['cobevt'])
        if args['fusion_method'] == 'where2comm':
            self.fusion_net = Where2commFusion(args['where2comm'])
        if args['fusion_method'] == 'who2com':
            self.fusion_net = Who2comFusion(args['who2com'])
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
        
        if 'message_extractor' in args and args['message_extractor'] == 'message_extractor':
            self.message_extractor = MessageExtractor(128, 2)
            print('message_extractor: message_extractor')
        if 'message_extractor' in args and args['message_extractor'] == 'message_extractorv2':
            self.message_extractor = MessageExtractorv2(128, 2)
            print('message_extractor: message_extractorv2')
        if 'message_extractor' in args and args['message_extractor'] == 'conv2d':
            self.message_extractor = nn.Conv2d(128, 2, kernel_size=1, padding=0)
            print('message_extractor: conv2d')
        if 'enhancer' in args and args['enhancer'] == 'enhance':
            self.enhancer = Enhancer(128, [8, 8], 4)
            print("use enhance")
        if 'enhancer' in args and args['enhancer'] == 'enhancev1':
            self.enhancer = Enhancerv1(128, [8, 8], 4)
            print("use enhancev1")
        if 'enhancer' in args and args['enhancer'] == 'enhancev2':
            self.enhancer = Enhancerv2(128, [8, 8], 4)
            print("use enhancev2")
        if 'enhancer' in args and args['enhancer'] == 'enhancev3':
            self.enhancer = Enhancerv3(128, [8, 8], 4)
            print("use enhancev3")
        if 'enhancer' in args and args['enhancer'] == 'enhancev4':
            self.enhancer = Enhancerv4(128, [8, 8], 4)
            print("use enhancev4")
        if 'enhancer' in args and args['enhancer'] == 'enhancev5':
            self.enhancer = Enhancerv5(128, [8, 8], 4)
            print("use enhancev5")
        if 'enhancer' in args and args['enhancer'] == 'enhancev6':
            self.enhancer = Enhancerv6(128, [8, 8], 4)
            print("use enhancev6")            
        if 'enhancer' in args and args['enhancer'] == 'enhancev7':
            self.enhancer = Enhancerv7(128, [8, 8], 4)
            print("use enhancev7")
        if 'enhancer' in args and args['enhancer'] == 'enhancev8':
            self.enhancer = Enhancerv8(128, [8, 8], 4)
            print("use enhancev8")        
        if 'enhancer' in args and args['enhancer'] == 'enhancev9':
            self.enhancer = Enhancerv9(128, [8, 8], 4)
            print("use enhancev9")
        if 'enhancer' in args and args['enhancer'] == 'enhancev10':
            self.enhancer = Enhancerv10(128, [8, 8], 4)
            print("use enhancev10") 
        if 'enhancer' in args and args['enhancer'] == 'enhancev11':
            self.enhancer = Enhancerv11(128, [8, 8], 4)
            print("use enhancev11") 
        if 'enhancer' in args and args['enhancer'] == 'enhancev12':
            self.enhancer = Enhancerv12(128, [8, 8], 4)
            print("use enhancev12")
        if 'enhancer' in args and args['enhancer'] == 'enhancev13':
            self.enhancer = Enhancerv13(128, [8, 8], 4)
            print("use enhancev13")   
        
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
        output_dict = {}
        agent_modality_list = data_dict['agent_modality_list'] 
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len'] 
        # print(agent_modality_list)

        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
            feature = eval(f"self.shrinker_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature

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
        for modality_name in agent_modality_list:
            feat_idx = counting_dict[modality_name]
            heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])
            counting_dict[modality_name] += 1

        heter_feature_2d = torch.stack(heter_feature_2d_list)
        
        if self.compress:
            heter_feature_2d = self.compressor(heter_feature_2d)

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

        """
        Feature Fusion (multiscale).

        we omit self.backbone's first layer.
        """
        
        
        gt_feature = heter_feature_2d
        if hasattr(self, 'message_extractor'):
            conditions = self.message_extractor(heter_feature_2d)
        else:
            conditions = self.cls_head(heter_feature_2d)
        
        if hasattr(self, 'condition_bias') and self.condition_bias:
            conditions_split = regroup(conditions, record_len)
            condi = []
            for b in range(len(record_len)):
                affine_matrix_ = affine_matrix[b,0][:conditions_split[b].shape[0]]
                feat_h, feat_w = conditions.shape[2], conditions.shape[3]
                angle_bias = get_angle_bias(self.angle_tabels, self.angle_bins, conditions, affine_matrix_, feat_h, feat_w).permute(0, 3, 1, 2).contiguous()
                distance_bias = get_distance_bias(self.distance_tabels, self.distance_bins, conditions, affine_matrix_, feat_h, feat_w).permute(0, 3, 1, 2).contiguous()
                conditions = torch.cat([conditions, angle_bias, distance_bias], dim=1)
                condi.append(conditions)
            conditions = torch.cat(condi, dim=0)

            

        gen_data_dict = self.diffcomm(heter_feature_2d, conditions, record_len)
        output_dict.update({'gt_feature': gt_feature,
                            'pred_feature': gen_data_dict['pred_feature']})
        
        note = 'm2_'
        # vis_bev(conditions[0].squeeze(0).detach().cpu().numpy(), type=note + 'ego_condi')
        # vis_bev(conditions[1].squeeze(0).detach().cpu().numpy(), type=note + 'cav_condi')
        # vis_bev(gt_feature[0].detach().cpu().numpy(), type='ego')
        # vis_bev(gen_data_dict['t1'].squeeze(0).detach().cpu().numpy(), type=note + 'ego_noisy_t1')
        # vis_bev(gen_data_dict['t2'].squeeze(0).detach().cpu().numpy(), type=note + 'ego_noisy_t2')
        # vis_bev(gt_feature[1].detach().cpu().numpy(), type='cav')
        # vis_bev(gen_data_dict['pred_feature'][0].detach().cpu().numpy(), type=note + 'ego_gen')
        # vis_bev(gen_data_dict['pred_feature'][1].detach().cpu().numpy(), type=note + 'cav_gen')
        
        heter_feature_2d = gen_data_dict['pred_feature']
 
        if len(heter_feature_2d.shape) == 3:
            heter_feature_2d = heter_feature_2d.unsqueeze(0) ## for the case of bs=1 and only ego
        if hasattr(self, 'enhancer'):
            heter_feature_2d = self.enhancer(heter_feature_2d, affine_matrix, record_len)
            if self.gmatch:
                gt_feature = self.enhancer(gt_feature, affine_matrix, record_len)
            
        fused_feature = self.fusion_net(heter_feature_2d, record_len, affine_matrix)
        if self.gmatch:
            fused_feature_T = self.fusion_net(gt_feature, record_len, affine_matrix)

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
                                'dir_preds_T': dir_preds_T,
                                'cls_preds_S': cls_preds,
                                'reg_preds_S': reg_preds,
                                'dir_preds_S': dir_preds})

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds,
                            'message': conditions})

        return output_dict




