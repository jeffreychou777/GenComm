CUDA_VISIBLE_DEVICES=2 python opencood/tools/inference.py \
    --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/m1_diffcomm_enhance_2025_03_28_06_07_00 \
    --range "102.4,51.2"

CUDA_VISIBLE_DEVICES=2 python opencood/tools/inference.py \
    --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/m1_diffcomm_halfmap_2025_03_29_15_10_35 \
    --range "102.4,51.2"