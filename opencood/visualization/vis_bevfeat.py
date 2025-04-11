import numpy as np
import matplotlib.pyplot as plt

def vis_bev(bev_feature, type):
    BEV_feature = bev_feature  # 生成示例数据

    # 在通道维度求和或平均
    BEV_sum = BEV_feature.sum(axis=0)  # 求和
    # 或者 BEV_avg = BEV_feature.mean(axis=0)  # 平均

    # 可视化
    plt.figure(figsize=(10, 5))
    plt.imshow(BEV_sum, cmap='hot', vmin=BEV_sum.min(), vmax=BEV_sum.max(), origin='lower')
    plt.axis('off')
    plt.colorbar()
    plt.savefig(f'/home/junfei.zhou/DATACENTER2/data/code/DiffComm/feat_vis_conv/{type}.png', dpi=500)
    plt.close()