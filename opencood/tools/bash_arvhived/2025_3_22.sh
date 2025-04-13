CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch  --nproc_per_node=2 --use_env opencood/tools/train_gmatch_ddp.py \
-y /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/hypes_yaml/opv2v/MoreModality/Diffcomm/grad_match/m1_diffcomm_gmatch_where2comm.yaml

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch  --nproc_per_node=2 --use_env opencood/tools/train_ddp.py \
-y /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/hypes_yaml/opv2v/MoreModality/Diffcomm/single_modality/m1_diffcomm_where2comm.yaml
