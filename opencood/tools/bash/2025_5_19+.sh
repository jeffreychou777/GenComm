CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference_w_noise.py \
    --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/codebook/m1m3_v2xvit

CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference_w_noise.py \
    --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/mpda/m1m3_v2xvit

CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference_w_noise.py \
    --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/diffcomm_align/m1m3_v2xvit

CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference_w_noise.py \
    --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/backalign/m1m3_v2xvit