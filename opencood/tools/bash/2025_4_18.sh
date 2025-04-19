CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch  \
    --nproc_per_node=2 --use_env opencood/tools/train_ddp.py \
    -y /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/hypes_yaml/opv2v/DiffComm/base_training/single/m1_v2xvit.yaml

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch  \
    --nproc_per_node=2 --use_env opencood/tools/train_ddp.py \
    -y /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/hypes_yaml/opv2v/DiffComm/base_training/single/m3_v2xvit.yaml