import sys
import runpy
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

os.chdir('/home/junfei.zhou/DATACENTER2/data/code/DiffComm')
# args = 'python -m lilab.multiview_scripts_new.s2_matpkl2ballpkl /mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/TPH2KOxWT/2022-06-16ball.matpkl --time 1 9 17 23 27'
# args = 'python -m lilab.metric_seg.s3_cocopkl_vs_cocopkl --gt_pkls /home/liying_lab/chenxinfeng/DATA/CBNetV2/data/rats_metric/te1/intense_pannel.cocopkl --pred_pkls /home/liying_lab/chenxinfeng/DATA/CBNetV2/data/rats_metric/te2/intense_pannel.cocopkl '

m1m3_pyramid_codebook = 'python opencood/tools/train.py -y None --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/codebook/m1m3_pyramid'
m1m3_pyramid_backalign = 'python opencood/tools/train.py -y None --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/backalign/m1m3_pyramid'

m1m2_att_mpda = 'python opencood/tools/train.py -y None --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/mpda/m1m2_att'
m1_v2xvit_diffcomm = 'python opencood/tools/train.py -y /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/hypes_yaml/opv2v/DiffComm/base_training/single_w_diffcomm/m1_v2xvit_diffcomm.yaml'
m1m3_pnpda = 'python opencood/tools/train.py -y None --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/pnpda'

m1m3_codebook_where2comm = 'python opencood/tools/train.py -y None --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/codebook/m1m3_w2c'

m1m4_diffcomm_align = 'python opencood/tools/train.py -y None --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/diffcomm_align/m1m4_v2xvit'

m0m3_stamp = 'python opencood/tools/train_adapter.py -y None --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/stamp/m0m3_att'
args = m0m3_stamp

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
