import sys
import runpy
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

os.chdir('/home/junfei.zhou/DATACENTER2/data/code/DiffComm')
# args = 'python -m lilab.multiview_scripts_new.s2_matpkl2ballpkl /mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/TPH2KOxWT/2022-06-16ball.matpkl --time 1 9 17 23 27'
# args = 'python -m lilab.metric_seg.s3_cocopkl_vs_cocopkl --gt_pkls /home/liying_lab/chenxinfeng/DATA/CBNetV2/data/rats_metric/te1/intense_pannel.cocopkl --pred_pkls /home/liying_lab/chenxinfeng/DATA/CBNetV2/data/rats_metric/te2/intense_pannel.cocopkl '

two_modalities = 'python opencood/tools/train.py --hypes_yaml /home/zjf/DATACENTER2/data/code/HEAL/opencood/hypes_yaml/opv2v/MoreModality/2_modality_end2end_training/lidar_camera_attfuse.yaml'
three_modalities = 'python opencood/tools/train.py --hypes_yaml  /home/zjf/DATACENTER2/data/code/HEAL/opencood/hypes_yaml/opv2v/MoreModality/3_modality_end2end_training/m1m2m3_attfuse.yaml'

m1_base = 'python opencood/tools/train.py -y None --model_dir opencood/logs/HEAL_m1_based/stage1/m1_base'

diffcomm_m1_base = 'python opencood/tools/train.py -y None --model_dir /home/zjf/DATACENTER2/data/code/HEAL/opencood/logs/DiffComm_m1_based/stage1/m1_base'
diffcomm_m2_base = 'python opencood/tools/train.py -y None --model_dir /home/zjf/DATACENTER2/data/code/HEAL/opencood/logs/DiffComm_m1_based/stage1/m2_base'
diffcomm_m3_base = 'python opencood/tools/train.py -y None --model_dir /home/zjf/DATACENTER2/data/code/HEAL/opencood/logs/DiffComm_m1_based/stage1/m3_base'
diffcomm_m4_base = 'python opencood/tools/train.py -y None --model_dir /home/zjf/DATACENTER2/data/code/HEAL/opencood/logs/DiffComm_m1_based/stage1/m4_base'

baseline_m1 = 'python opencood/tools/train.py -y None --model_dir /home/zjf/DATACENTER2/data/code/HEAL/opencood/logs/baseline/opv2v/att/m1'
baseline_m2 = 'python opencood/tools/train.py -y None --model_dir /home/zjf/DATACENTER2/data/code/HEAL/opencood/logs/baseline/opv2v/att/m2'

heal_m1based_m2_alignto_m1 = 'python opencood/tools/train.py -y None --model_dir /home/zjf/DATACENTER2/data/code/HEAL/opencood/logs/HEAL_m1_based/stage2/m2_alignto_m1' 

heal_m1based_final_infer = 'python opencood/tools/inference_heter_in_order.py --model_dir opencood/logs/HEAL_m1_based/final_infer'

m1m3_e2e = 'python opencood/tools/train.py -y /home/zjf/DATACENTER2/data/code/HEAL/opencood/hypes_yaml/opv2v/MoreModality/2_modality_end2end_training/m1m3_attfuse.yaml'

m1_diffcomm_infer = 'python opencood/tools/inference.py --model_dir /home/zjf/DATACENTER2/data/code/HEAL/opencood/logs/m1_diffcomm_2025_03_12_13_01_19'
heal_diffcomm_m3_alignto_m1 = 'python opencood/tools/train.py -y None --model_dir /home/zjf/DATACENTER2/data/code/HEAL/opencood/logs/DiffComm_m1_based/stage2/m3_alignto_m1_only_mess_extrc'

m1m3_diffcomm_infer = 'python opencood/tools/inference.py --model_dir /home/zjf/DATACENTER2/data/code/HEAL/opencood/logs/DiffComm_m1_based/infer/m1m3_infer'

m1m3_infer = 'python opencood/tools/inference.py --model_dir /home/zjf/DATACENTER2/data/code/HEAL/opencood/logs/DiffComm_m1_based/infer/m1m3_wo_diffcomm_train_infer'

gmatch = 'python opencood/tools/train_gmatch.py -y /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/hypes_yaml/opv2v/MoreModality/Diffcomm/grad_match/m1_diffcomm_gmatch.yaml'
gmatch_v2xvit = 'python opencood/tools/train_gmatch.py -y /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/hypes_yaml/opv2v/MoreModality/Diffcomm/grad_match/m1_diffcomm_gmatch_v2xvit.yaml'

message_extract = 'python opencood/tools/train.py -y /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/hypes_yaml/opv2v/MoreModality/Diffcomm/message_extractor/m1_diffcomm_mess_extract.yaml'

v2xvit = 'python opencood/tools/train.py -y /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/hypes_yaml/opv2v/LiDAROnly/lidar_v2xvit.yaml'
m1_diffcomm_v2xvit = 'python opencood/tools/train.py -y /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/hypes_yaml/opv2v/MoreModality/Diffcomm/verify_v2xvit/m1_diffcomm.yaml'

vis_cls_head = 'python opencood/tools/inference.py --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/m1_diffcomm_2025_03_25_07_47_43'
vis_conv2d = 'python opencood/tools/inference.py --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/m1_diffcomm_mess_extrac_2025_03_26_01_23_11'

infer = 'python opencood/tools/inference.py --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/m1_diffcomm_enhance_2025_03_27_12_01_36'
enhance = 'python opencood/tools/train.py -y /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/hypes_yaml/opv2v/MoreModality/Diffcomm/message_extractor/m1_diffcomm_enhance.yaml'

m1_diffcomm_halfmap = 'python opencood/tools/train.py -y /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/hypes_yaml/opv2v/MoreModality/Diffcomm/verify_att/m1_diffcomm.yaml'
half_range_infer = 'python opencood/tools/inference.py \
    --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/m1_diffcomm_gmatch_2025_03_28_15_22_09 \
    --range 102.4,51.2'

m1_diffcomm_ehance_woms = 'python opencood/tools/train.py -y /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/hypes_yaml/opv2v/MoreModality/Diffcomm/message_extractor/m1_diffcomm_enhance_woms.yaml'

angle_bias_vis = 'python opencood/tools/inference.py --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/m1_diffcomm_enhance_2025_04_03_03_14_44 --range 102.4,51.2'
distance_bias_vis = 'python opencood/tools/inference.py --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/m1_diffcomm_enhance_2025_04_03_23_58_06 --range 102.4,51.2'

m1_diffcomm_diffusion = 'python opencood/tools/train.py -y /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/hypes_yaml/opv2v/MoreModality/Diffcomm/diffusion/m1_diffcomm_diffusion.yaml'
m1_diffcomm_diffusion_infer = 'python opencood/tools/inference.py --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/m1_diffcomm_diffusion_2025_04_05_04_55_04'

m2_vis = 'python opencood/tools/inference.py --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/m2_diffcomm_2025_04_06_04_17_22 --range 51.2,51.2'

diffcomm_m1based_m2_alignto_m1 = 'python opencood/tools/train.py -y None --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/stage2/m2_alignto_m1'
heal_diffcomm_m2_alignto_m1 = 'python opencood/tools/train.py -y None --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/HEAL_m1_based/stage2/m2_alignto_m1'
diffcomm_m2_alignto_m1_vis = 'python opencood/tools/inference.py --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/stage2/m2_alignto_m1'

m1_att_diffcomm = 'python opencood/tools/train_gmatch.py -y /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/hypes_yaml/opv2v/DiffComm/base_training/single_w_diffcomm/m1_att_diffcomm.yaml'
m3_att_diffcomm = 'python opencood/tools/train_gmatch.py -y /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/hypes_yaml/opv2v/DiffComm/base_training/single/m3_att.yaml'
m3_att_wo_diffcomm = 'python opencood/tools/train.py -y /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/hypes_yaml/opv2v/DiffComm/base_training/single/m3_att.yaml'
m1m3_att_mpda = 'python opencood/tools/train.py -y None --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/mpda/m1m3'

hdf5_test = 'python opencood/tools/inference.py --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/m2_base_att_wo_diffcomm_2025_04_12_03_19_03 --range 51.2,51.2'
m1m3_att_infer = 'python opencood/tools/inference.py --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/infer/m1m3_att_new'
m1m3_fcooper_infer = 'python opencood/tools/inference.py --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/infer/m1m3_fcooper'
m1m3_att_old = 'python opencood/tools/inference.py --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/infer/m1m3_att_old'
m1m3_att_new = 'python opencood/tools/inference.py --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/infer/m1m3_att_newer'
m1m3_unknown = 'python opencood/tools/inference.py --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/m3_alignto_m1_infer_wo_align'
m1m2_att = 'python opencood/tools/inference.py --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/infer/m1m2_att'

m2_alignto_m1 = 'python opencood/tools/inference.py --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/m2_alignto_m1_infer_wo_align'
m1m3_pyramid_infer = 'python opencood/tools/inference.py --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/infer/m1m3_pyramid'
m3_att_new = 'python opencood/tools/train.py -y /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/hypes_yaml/opv2v/DiffComm/base_training/single/m3_att.yaml'

m1m3_pnpda = 'python opencood/tools/train.py -y None --model_dir  /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/pnpda'
m1m3_where2comm = 'python opencood/tools/inference.py --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/infer/m1m3_where2comm'
m1m3_codebook = 'python opencood/tools/train.py -y None --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/codebook/m1m3_att'

args = m1m3_codebook


# args = 'python opencood/tools/inference.py --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/m1_diffcomm_enhance_2025_03_30_01_47_16'

# args = 'python opencood/tools/train_gmatch.py -y /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/hypes_yaml/opv2v/DiffComm/base_training/verify/m1_att_diffcomm_dropout.yaml'

# args = 'python test.py 5 7'
args = args.split()
if args[0] == 'python':
    """pop up the first in the args""" 
    args.pop(0)

if args[0] == '-m':
    """pop up the first in the args"""
    args.pop(0)
    fun = runpy.run_module
else:
    fun = runpy.run_path

sys.argv.extend(args[1:])

fun(args[0], run_name='__main__')
