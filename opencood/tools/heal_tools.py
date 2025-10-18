# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import os
import sys
from collections import OrderedDict
import glob
import re

def get_model_path_from_dir(model_dir):
    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            raise "No checkpoint!"
        
        return os.path.join(save_dir, f'net_epoch{initial_epoch_}.pth')

    file_list = glob.glob(os.path.join(model_dir, 'net_epoch_bestval_at*.pth'))

    if len(file_list):
        assert len(file_list) == 1
        model_path = file_list[0]
    else:
        model_path = findLastCheckpoint(model_dir)

    print(f"find {model_path}.")
    
    return model_path


def rename_to_new_version(checkpoint_path):
    # stage1 model to new vesrion
    # 加载 checkpoint
    old_state_dict = torch.load(checkpoint_path)

    # 创建一个新的字典，用于保存重命名后的键值对
    new_state_dict = OrderedDict()

    # 遍历旧的 state_dict，将所有的键进行重命名，然后保存到新的字典中
    for key in old_state_dict:
        # 将 'model.model' 替换为 'channel_align.model'
        new_key = key.replace('model.model', 'channel_align.model')
        new_key = new_key.replace('model.warpnet', 'warpnet')
        new_state_dict[new_key] = old_state_dict[key]


    # 保存新的 checkpoint
    torch.save(new_state_dict, checkpoint_path)
    torch.save(old_state_dict, checkpoint_path.replace(".pth", ".pth.oldversion"))

def remove_m4_trunk(checkpoint_path):
    # 加载 checkpoint
    old_state_dict = torch.load(checkpoint_path)

    # 创建一个新的字典，用于保存重命名后的键值对
    new_state_dict = OrderedDict()

    # 遍历旧的 state_dict，将所有的键进行重命名，然后保存到新的字典中
    for key in old_state_dict:
        if key.startswith("encoder_m4.camencode.trunk") or \
            key.startswith('encoder_m4.camencode.final_conv') or \
            key.startswith("encoder_m4.camencode.layer3"):
            continue

        new_state_dict[key] = old_state_dict[key]

    # 保存新的 checkpoint
    torch.save(new_state_dict, checkpoint_path)
    torch.save(old_state_dict, checkpoint_path.replace(".pth", ".pth.oldversion"))

def merge_dict(single_model_dict, stage1_model_dict):
    merged_dict = OrderedDict()
    single_keys = set(single_model_dict.keys())
    if "*fusion_net*" in single_keys:
        print('fusion_net in single_model_dict')
    stage1_keys = set(stage1_model_dict.keys())
    symm_diff_set = single_keys & stage1_keys
    overlap_module = set([key.split(".")[0] for key in symm_diff_set])
    print("=======Overlap modules in two checkpoints=======")
    print(*overlap_module, sep="\n")
    for param in symm_diff_set:
        if not torch.equal(single_model_dict[param], stage1_model_dict[param]):
            print(f"[WARNING]: Different param in {param}")
    print("================================================")

    for key in single_model_dict:
        # remove keys like 'layers_m4.resnet.layer2.0.bn1.bias' / 'cls_head_m4.weight' / 'shrink_conv_m4.weight'
        # from single_model_dict
        if 'layers_m' in key or 'head_m' in key or 'shrink_conv_m' in key: 
            print(f"Pass {key}")
            continue
        merged_dict[key] = single_model_dict[key]

    for key in stage1_keys:
        merged_dict[key] = stage1_model_dict[key]

    return merged_dict

def merge_dict_diffcomm(single_model_dict, stage1_model_dict):
    '''
    diffcomm means keeps all modality's backbone and encoder, 
    fusion_net, and head, message extractor, shrink_head
    '''
    
    merged_dict = OrderedDict()
    
    single_keys = set(single_model_dict.keys())
    stage1_keys = set(stage1_model_dict.keys())
    # single_modality = next((k for k in single_keys if "backbone" in k), None).split(".")[0].split("_")[1]
    # stage1_modality = next((k for k in stage1_keys if "backbone" in k), None).split(".")[0].split("_")[1]
    # single_model_dict['message_extractor_' + single_modality + '.weight'] \
    #     = single_model_dict['cls_head.weight']
    # single_model_dict['message_extractor_' + single_modality + '.bias'] \
    #     = single_model_dict['cls_head.bias']
    # stage1_model_dict['message_extractor_' + stage1_modality + '.weight'] \
    #     = stage1_model_dict['cls_head.weight']
    # stage1_model_dict['message_extractor_' + stage1_modality + '.bias'] \
    #     = stage1_model_dict['cls_head.bias']

    single_keys = set(single_model_dict.keys())
    stage1_keys = set(stage1_model_dict.keys())
    symm_diff_set = single_keys & stage1_keys
    overlap_module = set([key.split(".")[0] for key in symm_diff_set])
    print("=======Overlap modules in two checkpoints=======")
    print(*overlap_module, sep="\n")
    for param in symm_diff_set:
        if not torch.equal(single_model_dict[param], stage1_model_dict[param]):
            print(f"[WARNING]: Different param in {param}")
    print("================================================")

    for key in single_model_dict:
        # remove keys like 'layers_m4.resnet.layer2.0.bn1.bias' / 'cls_head_m4.weight' / 'shrink_conv_m4.weight'
        # from single_model_dict
        if 'layers_m' in key or 'head_m' in key or 'shrink_conv_m' in key: 
            print(f"Pass {key}")
            continue
        merged_dict[key] = single_model_dict[key]

    for key in stage1_keys:
        merged_dict[key] = stage1_model_dict[key]

    return merged_dict
    
    
def merge_and_save(single_model_dir, stage1_model_dir, output_model_dir):
    single_model_path = get_model_path_from_dir(single_model_dir)
    stage1_model_path = get_model_path_from_dir(stage1_model_dir)
    single_model_dict = torch.load(single_model_path, map_location='cpu')
    stage1_model_dict = torch.load(stage1_model_path, map_location='cpu')
    merged_dict = merge_dict(single_model_dict, stage1_model_dict)
    
    output_model_path = os.path.join(output_model_dir, 'net_epoch1.pth')
    torch.save(merged_dict, output_model_path)
    
def merge_and_save_diffcomm(single_model_dir, stage1_model_dir, output_model_dir, dair_flag=False):
    if dair_flag:
        single_model_dict = change_modality_key_name(single_model_dir)
    else:
        single_model_path = get_model_path_from_dir(single_model_dir)
        single_model_dict = torch.load(single_model_path, map_location='cpu')
    stage1_model_path = get_model_path_from_dir(stage1_model_dir)
    stage1_model_dict = torch.load(stage1_model_path, map_location='cpu')
    
    merged_dict = merge_dict_diffcomm(single_model_dict, stage1_model_dict)
    
    output_model_path = os.path.join(output_model_dir, 'net_epoch1.pth')
    torch.save(merged_dict, output_model_path)

def merge_dict_mpda(path1, path2):
    dict1 = torch.load(get_model_path_from_dir(path1), map_location='cpu')   #ego
    dict2 = torch.load(get_model_path_from_dir(path2), map_location='cpu')
    
    merged_dict = OrderedDict()
    
    dict1_keys = set(dict1.keys())
    dict2_keys = set(dict2.keys())
    
    symm_diff_set = dict1_keys & dict2_keys
    overlap_module = set([key.split(".")[0] for key in symm_diff_set])
    print("=======Overlap modules in two checkpoints=======")
    print(*overlap_module, sep="\n")
    for param in symm_diff_set:
        if not torch.equal(dict2[param], dict1[param]):
            print(f"[WARNING]: Different param in {param}")
    print("================================================")

    for key in dict2:
        # remove keys like 'layers_m4.resnet.layer2.0.bn1.bias' / 'cls_head_m4.weight' / 'shrink_conv_m4.weight'
        # from single_model_dict
        if 'layers_m' in key or 'head_m' in key or 'shrink_conv_m' in key: 
            print(f"Pass {key}")
            continue
        merged_dict[key] = dict2[key]

    for key in dict1_keys:
        merged_dict[key] = dict1[key]

    return merged_dict

def merge_and_save_final(aligned_model_dir_list, output_model_dir):
    """
    aligned_model_dir_list:
        e.g. [m2_ALIGNTO_m1_model_dir, m3_ALIGNTO_m1_model_dir, m4_ALIGNTO_m1_model_dir, m1_collaboration_base_dir]

    output_model_dir:
        model_dir.
    """
    final_dict = OrderedDict()
    for aligned_model_dir in aligned_model_dir_list:
        aligned_model_path = get_model_path_from_dir(aligned_model_dir)
        model_dict = torch.load(aligned_model_path, map_location='cpu')
        final_dict = merge_dict(final_dict, model_dict)

    output_model_path = os.path.join(output_model_dir, 'net_epoch1.pth')
    torch.save(final_dict, output_model_path)

def add_suffix_to_keys(model_dict, suffix):
    """
    Add suffix to keys in model_dict.
    """
    for key in model_dict.keys():
        if key.startswith('message_extractor.'):
            new_key = key.replace('message_extractor.', f'message_extractor_{suffix}.')
            model_dict[new_key] = model_dict[key]
    return model_dict

def add_suffix_to_keys_save(log_path, suffix, save_path):
    """
    Add suffix to keys in model_dict.
    """
    model_path = get_model_path_from_dir(log_path)

    model_dict = torch.load(model_path, map_location='cpu')
    for key in list(model_dict.keys()):
        if key.startswith('reg_head.'):
            new_key = key.replace('reg_head.', f'reg_head_{suffix}.')
            model_dict[new_key] = model_dict.pop(key)
        if key.startswith('cls_head.'):
            new_key = key.replace('cls_head.', f'cls_head_{suffix}.')
            model_dict[new_key] = model_dict.pop(key)
        if key.startswith('dir_head.'):
            new_key = key.replace('dir_head.', f'dir_head_{suffix}.')
            model_dict[new_key] = model_dict.pop(key)
    torch.save(model_dict, os.path.join(save_path,'net_epoch1.pth'))

def change_modality_key_name(log_path,):
    """
    Change 'm1' to 'mx' in model_dict.
    """
    model_path = get_model_path_from_dir(log_path)
    model_dict = torch.load(model_path, map_location='cpu')
    for key in list(model_dict.keys()):
        if 'm3' in key:
            new_key = key.replace('m3', 'm0')
            model_dict[new_key] = model_dict.pop(key)
            
    torch.save(model_dict, os.path.join(model_path))
    
    return model_dict


if __name__ == "__main__":
    func = sys.argv[1]
    if func == 'rename_to_new_version':
        checkpoint_path = sys.argv[2]
        rename_to_new_version(checkpoint_path)
    elif func == 'remove_m4_trunk':
        checkpoint_path = sys.argv[2]
        remove_m4_trunk(checkpoint_path)
    elif func == 'add_suffix_to_keys_save':
        log_path = sys.argv[2]
        suffix = sys.argv[3]
        save_path = sys.argv[4]
        add_suffix_to_keys_save(log_path, suffix, save_path)
    elif func == 'merge':
        single_model_dir = sys.argv[2]
        stage1_model_dir = sys.argv[3]
        output_model_dir = sys.argv[4]
        merge_and_save(single_model_dir, stage1_model_dir, output_model_dir)
    elif func == 'merge_and_save': 
        merge_and_save_final(sys.argv[2:-1], sys.argv[-1])
    else:
        raise "This function not implemented"
    
    # change_modality_key_name(log_path='/home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DAIR_m3_attfuse_wo_diffcomm_2025_04_28_06_44_56')
    
    
    # single_model_dir = '/home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/HeterBaselin_V2XReal_lidar_attfuse_m4_2025_07_28_00_19_41'
    # stage1_model_dir = '/home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/HeterBaselin_V2XReal_lidar_attfuse_m1_2025_07_27_03_42_31'
    # output_model_dir = '/home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/mpda/V2XReal_m1m4_att'
    # merge_and_save_diffcomm(single_model_dir, stage1_model_dir, output_model_dir, dair_flag=False)
    
    
    # add_suffix_to_keys_save(log_path = '/DATACENTER3/home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/stamp/opv2v-h/m1m3_v2xvit_infer',
    #                         suffix = 'm1',
    #                         save_path='/DATACENTER3/home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/stamp/opv2v-h/m1m3_v2xvit_infer')
    
    # m0_dir = '/DATACENTER3/home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/stamp/opv2v-h/m0_v2xvit_ckp'
    # m1_dir = '/DATACENTER3/home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/stamp/v2xreal/m1_ckp'
    # m2_dir = 'DATACENTER2/data/code/DiffComm/opencood/logs/m2_base_v2xvit_wo_diffcomm_2025_04_26_07_43_18'
    # m3_dir = '/DATACENTER3/home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/stamp/v2xreal/m0m3_att'
    # m4_dir = '/DATACENTER3/home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/stamp/v2xreal/m0m4_att'
    # infer = '/DATACENTER3/home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/stamp/v2xreal/infer'

    # dir_list = [m0_dir, m1_dir] ## put ego_dir as the last
    # output_model_dir = '/DATACENTER3/home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/stamp/opv2v-h/m0m2_v2xvit'
    # merge_and_save_final(dir_list, output_model_dir)
    
    # m1_m2_dir = '/DATACENTER3/home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/stamp/dair/m0m1_att'
    # m1_m3_dir = '/DATACENTER3/home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/stamp/dair/m0m3_att'
    # m1_m4_dir = '/home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/backalign/V2XReal_m1m4_att'

    # m1_dir = '/DATACENTER3/home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/stamp/opv2v-h/m1_v2xvit_ckp'
    # m0_m1_dir = '/DATACENTER3/home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/stamp/opv2v-h/m0m1_v2xvit'
    # m0_m2_dir = '/DATACENTER3/home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/stamp/opv2v-h/m0m2_v2xvit'      
    # dir_list = [m0_m1_dir, m0_m2_dir, m1_dir]
    # output_model_dir = '/DATACENTER3/home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/stamp/opv2v-h/m1m2_v2xvit_infer'
    # merge_and_save_final(dir_list, output_model_dir)
    
    # change_modality_key_name(log_path = '/DATACENTER3/home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/stamp/opv2v-h/m0_v2xvit_ckp')
    