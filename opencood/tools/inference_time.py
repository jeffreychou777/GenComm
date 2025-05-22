import torch
import time
from opencood.models.diffcomm_modules.cond_diff import DiffComm

# 模型配置
config = {
    
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

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DiffComm(config).to(device)
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
