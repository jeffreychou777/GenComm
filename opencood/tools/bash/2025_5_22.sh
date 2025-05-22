CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_heter_in_order.py \
    --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/stamp/infer_my \
    --use_cav [3] --no_score

CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_heter_in_order.py \
    --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/diffcomm_align/m1m2m3m4_att_infer \
    --use_cav [3] --no_score

CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_heter_in_order.py \
    --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/codebook/m1m2_att \
    --use_cav [3] --no_score