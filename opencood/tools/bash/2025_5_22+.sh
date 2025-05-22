CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference_heter_in_order.py \
    --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/backalign/m1m2_att \
    --use_cav [3] --no_score

CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference_heter_in_order.py \
    --model_dir /home/junfei.zhou/DATACENTER2/data/code/DiffComm/opencood/logs/DiffComm/mpda/m1m2_att \
    --use_cav [3] --no_score
