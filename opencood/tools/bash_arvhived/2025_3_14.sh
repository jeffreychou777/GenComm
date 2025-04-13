CUDA_VISIBLE_DEVICES=3 python opencood/tools/train.py -y None --model_dir /home/zjf/DATACENTER2/data/code/HEAL/opencood/logs/DiffComm_m1_based/stage2/m3_alignto_m1

CUDA_VISIBLE_DEVICES=3 python opencood/tools/train.py -y /home/zjf/DATACENTER2/data/code/HEAL/opencood/hypes_yaml/opv2v/MoreModality/Diffcomm/single_modality/m1_diffcomm.yaml

CUDA_VISIBLE_DEVICES=3 python opencood/tools/train.py -y /home/zjf/DATACENTER2/data/code/HEAL/opencood/hypes_yaml/opv2v/MoreModality/Diffcomm/single_modality/m1_wo_diffcomm.yaml