CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_w_noise.py \
    --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/codebook/DAIR_m1m3_att

CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_w_noise.py \
    --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/backalign/DAIR_m1m3_att

CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_w_noise.py \
    --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/diffcomm_align/DAIR_m1m3_att

CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_w_noise.py \
    --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/mpda/DAIR_m1m3_att