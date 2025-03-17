import sys
import runpy
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

os.chdir('/home/zjf/DATACENTER2/data/code/HEAL')
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

args = heal_diffcomm_m3_alignto_m1


# args = 'python opencood/tools/train.py -y /home/zjf/DATACENTER2/data/code/HEAL/opencood/hypes_yaml/opv2v/MoreModality/Diffcomm/verify/m1_diffcomm.yaml'

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
