CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py -y /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/hypes_yaml/opv2v/MoreModality/Diffcomm/verify_v2xvit/m1_diffcomm.yaml

CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py -y /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/hypes_yaml/opv2v/MoreModality/Diffcomm/verify_v2xvit/m1_wo_diffcomm.yaml

CUDA_VISIBLE_DEVICES=0 python opencood/tools/train_gmatch.py -y /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/hypes_yaml/opv2v/MoreModality/Diffcomm/grad_match/m1_diffcomm_gmatch_v2xvit.yaml