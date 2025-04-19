import torch
import numpy as np
import matplotlib.pyplot as plt

def vis_bev(bev_feature, type, min=None, max=None, normalize=False):
    BEV_feature = bev_feature  # 生成示例数据
    if normalize:
        BEV_feature = (BEV_feature - BEV_feature.min()) / (BEV_feature.max() - BEV_feature.min())
    # 在通道维度求和或平均
    BEV_sum = BEV_feature.sum(axis=0)  # 求和
    BEV_avg = BEV_feature.mean(axis=0)  # 平均

    Bev = BEV_avg
    
    vmin = Bev.min() if min is None else min
    vmax = Bev.max() if max is None else max
    # 可视化
    plt.figure(figsize=(10, 5))
    plt.imshow(Bev, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
    plt.axis('off')
    plt.colorbar()
    plt.savefig(f'/home/junfei.zhou/DATACENTER2/data/code/DiffComm/feat_vis_m1m3/{type}.png', dpi=500)
    plt.close()


def visualize_feature_maps(feature_map: torch.Tensor, 
                            mode: str = 'mean', 
                            channel: int = 0,
                            cmap: str = 'viridis',
                            figsize=(10, 4),
                            save_path: str = None):
    """
    可视化两个特征图（Ego & CAV）在统一色阶下的热图展示。

    Args:
        feature_map (Tensor): 输入 Tensor, shape = [2, C, H, W]
        mode (str): 'mean' 表示通道平均, 'single' 表示指定通道
        channel (int): 如果 mode='single'，要可视化的通道编号
        cmap (str): 颜色映射，比如 'viridis', 'plasma', 'jet'
        figsize (tuple): 图像尺寸
        save_path (str): 若指定路径，将保存图像
    """

    assert feature_map.ndim == 4 and feature_map.shape[0] == 2, "输入形状应为 [2, C, H, W]"

    if mode == 'mean':
        ego_feat = feature_map[0].mean(dim=0).cpu().numpy()
        cav_feat = feature_map[1].mean(dim=0).cpu().numpy()
        title_suffix = " (Mean over Channels)"
    elif mode == 'single':
        ego_feat = feature_map[0, channel].cpu().numpy()
        cav_feat = feature_map[1, channel].cpu().numpy()
        title_suffix = f" (Channel {channel})"
    else:
        raise ValueError("mode 必须是 'mean' 或 'single'")

    # 统一色阶
    vmin = min(ego_feat.min(), cav_feat.min())
    vmax = max(ego_feat.max(), cav_feat.max())

    # 可视化
    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    plt.title("Ego Feature" + title_suffix)
    plt.imshow(ego_feat, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("CAV Feature" + title_suffix)
    plt.imshow(cav_feat, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
