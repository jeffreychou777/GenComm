import torch
import torch.nn as nn
import time
from opencood.models.diffcomm_modules.cond_diff import DiffComm
from opencood.models.mpda_modules.classfier import DAImgHead
from opencood.models.mpda_modules.resizer import LearnableResizer
from opencood.models.mpda_modules.wg_fusion_modules import CrossDomainFusionEncoder
from opencood.models.sub_modules.codebook import UMGMQuantizer
from opencood.models.stamp_modules.adapter import Adapter, Reverter


# 模型配置
diffcomm_config = {
        "model": {
            "embed_dim": 129,
            "condition_channels": 1,
            "in_channels": 128,
            "out_ch": 128,
            "ch": 8,
            "ch_mult": [1, 1],
            "num_res_blocks": 2,
            "attn_resolutions": [16],
            "dropout": 0.0,
            "resamp_with_conv": True
        },
        "diffusion": {
            "beta_schedule": "linear",
            "beta_start": 0.0005,
            "beta_end": 0.02,
            "num_diffusion_timesteps": 3
        }

}

mpda_config = {  "resizer": {
        "input_channel": 128,
        "output_channel": 128,
        "wg_att": {
            "input_dim": 128,
            "mlp_dim": 256,
            "window_size": 2,
            "dim_head": 32,
            "drop_out": 0.1,
            "depth": 1
        },
        "residual": {
            "input_dim": 128,
            "depth": 2
        }
    },
    "classifier": 128,
    "cdt": {
        "input_dim": 128,
        "window_size": 2,
        "dim_head": 32,
        "heads": 16,
        "depth": 1
    }}


codebook_config = {
    "seg_num": 2,
    "dict_size": 16,
    "r": 3,
    "channel": 128
}
stamp_config = {
    "adapter": {
        "core_method": "adapterconvnext",
        "args": {
            "in_channels": 128,
            "out_channels": 128,
            "in_cav_lidar_range": [-102.4, -51.2, -3, 102.4, 51.2, 1],
            "out_cav_lidar_range": [-102.4, -51.2, -3, 102.4, 51.2, 1],
            "in_feature_shape": [128, 256],
            "out_feature_shape": [128, 256],
            "submodule_args": {
                "num_of_blocks": 3,
                "dim": 64
            }
        }
    },
    "reverter": {
        "core_method": "adapterconvnext",
        "args": {
            "in_channels": 128,
            "out_channels": 128,
            "in_cav_lidar_range": [-102.4, -51.2, -3, 102.4, 51.2, 1],
            "out_cav_lidar_range": [-102.4, -51.2, -3, 102.4, 51.2, 1],
            "in_feature_shape": [128, 256],
            "out_feature_shape": [128, 256],
            "submodule_args": {
                "num_of_blocks": 3,
                "dim": 64
            }
        }
    }
}

def evaluate_diffcomm():
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffComm(diffcomm_config).to(device)
    model.eval()

    # 构造模拟输入（请根据你实际输入修改）

    x = torch.randn((1, 128, 128, 256)).to(device)
    condition = torch.randn((2, 1 , 128, 256)).to(device)
    record_len = torch.tensor([2]).to(device)

    # GPU预热
    for _ in range(10):
        with torch.no_grad():
            _ = model(x, condition, record_len)

    # 测试推理时间：运行100次
    repeat = 100
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(repeat):
            _ = model(x, condition, record_len)

    torch.cuda.synchronize()
    end_time = time.time()

    avg_infer_time_ms = (end_time - start_time) * 1000 / repeat
    print(f"Average inference time over {repeat} runs: {avg_infer_time_ms:.3f} ms")

def evaluate_mpda():
    
    return 0

def evaluate_codebook():
    args = codebook_config
    channel = args['channel'] if 'channel' in args else 128
    p_rate = 0.0
    seg_num = args['seg_num']
    dict_size = [args['dict_size'], args['dict_size'], args['dict_size']]
    
    coodbook = UMGMQuantizer(channel, seg_num, dict_size, p_rate,
                        {"latentStageEncoder": lambda: nn.Linear(channel, channel), "quantizationHead": lambda: nn.Linear(channel, channel),
                        "latentHead": lambda: nn.Linear(channel, channel), "restoreHead": lambda: nn.Linear(channel, channel),
                        "dequantizationHead": lambda: nn.Linear(channel, channel), "sideHead": lambda: nn.Linear(channel, channel)})
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = coodbook.to(device)
    model.eval()
    
    x = torch.randn((4, 128, 128, 256)).to(device)
    x = x.permute(0, 2, 3, 1).contiguous().view(-1, channel)


    # GPU预热
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)

    # 测试推理时间：运行100次
    repeat = 100
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(repeat):
            _ = model(x)

    torch.cuda.synchronize()
    end_time = time.time()

    avg_infer_time_ms = (end_time - start_time) * 1000 / repeat
    print(f"Average inference time over {repeat} runs: {avg_infer_time_ms:.3f} ms")
    
    return 0

def evaluate_mpda():
    args = mpda_config
    resizer = LearnableResizer(args['resizer'])
    cdt = CrossDomainFusionEncoder(args['cdt'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resizer = resizer.to(device).eval()
    cdt = cdt.to(device).eval()
    
    ego_feature = torch.randn((1, 128, 128, 256)).to(device)
    cav_feature = torch.randn((4, 128, 128, 256)).to(device)
    # GPU预热
    for _ in range(10):
        with torch.no_grad():
            cav_feature = resizer(ego_feature, cav_feature)
            ego_copy = ego_feature.repeat(cav_feature.shape[0], 1, 1, 1)
            cav_feature = cdt(ego_copy, cav_feature)

    # 测试推理时间：运行100次
    repeat = 100
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(repeat):
            cav_feature = resizer(ego_feature, cav_feature)
            ego_copy = ego_feature.repeat(cav_feature.shape[0], 1, 1, 1)
            cav_feature = cdt(ego_copy, cav_feature)

    torch.cuda.synchronize()
    end_time = time.time()

    avg_infer_time_ms = (end_time - start_time) * 1000 / repeat
    print(f"Average inference time over {repeat} runs: {avg_infer_time_ms:.3f} ms")
    

def evaluate_stamp():
    adapter = Adapter(stamp_config['adapter'])
    reverter = Reverter(stamp_config['reverter'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = adapter.to(device).eval()
    reverter = reverter.to(device).eval()
    
    x = torch.randn((4, 128, 128, 256)).to(device)
    # GPU预热
    for _ in range(10):
        with torch.no_grad():
            x = adapter(x)
            x = reverter(x)

    # 测试推理时间：运行100次
    repeat = 100
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(repeat):
            x = adapter(x)
            x = reverter(x)

    torch.cuda.synchronize()
    end_time = time.time()

    avg_infer_time_ms = (end_time - start_time) * 1000 / repeat
    print(f"Average inference time over {repeat} runs: {avg_infer_time_ms:.3f} ms")
if __name__ == "__main__":
    # evaluate_codebook()
    # evaluate_mpda()
    evaluate_stamp()