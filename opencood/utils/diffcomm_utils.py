import torch
import math


def get_angle_bias(tabels, bins, x, affine_matrix, H, W):
    """
    计算基于角度的attention bias (范围1-2)
    """
    device = x.device
    # x = x.view(-1, H*W, x.size(-1))  # (B, H, W, C)
    # B, N, C = x.shape
    num_cav = affine_matrix.size(0)
    
    # 生成坐标网格
    h_idx = torch.arange(H, device=device)
    w_idx = torch.arange(W, device=device)
    grid_y, grid_x = torch.meshgrid(h_idx, w_idx, indexing='ij')
    token_coords = torch.stack([
        grid_x.reshape(-1),
        grid_y.reshape(-1)
    ], dim=-1).float()  # (H*W, 2)
    
    # 计算ego位置
    img_center = torch.tensor([[W/2], [H/2]], device=device).float()
    img_center = img_center.unsqueeze(0).expand(num_cav, -1, -1)
    affine_matrix = affine_matrix.float()
    ego_pos = torch.bmm(affine_matrix[:, :2, :2], img_center) + affine_matrix[:, :2, 2:3]
    ego_pos = ego_pos.squeeze(-1)  # (B, 2)
    
    # 计算相对位置和角度
    token_coords = token_coords.unsqueeze(0)  # (1, N, 2)
    ego_pos = ego_pos.unsqueeze(1)  # (B, 1, 2)
    rel_pos = token_coords - ego_pos  # (B, N, 2)
    
    angles = torch.atan2(rel_pos[..., 1], rel_pos[..., 0])  # (B, N)
    angle_normalized = (angles + math.pi) / (2 * math.pi)
    angle_bins = (angle_normalized * (bins-1)).long()  # (B, N)
    
    # 获取角度attention bias (1-2范围)
    angle_bias = tabels[angle_bins] # (B, N, num_heads)
    
    # reshape为attention格式
    angle_bias = angle_bias.view(-1, H, W).contiguous().unsqueeze(-1)  # (B, N, 1)
    
    return angle_bias
    
def get_distance_bias(tabels, bins, x, affine_matrix, H, W):
    """
    计算基于距离的attention bias (范围1-2)
    
    Args:
        x: tensor of shape (B, N, C)
        affine_matrix: tensor of shape (B, 2, 3)
        H: feature height
        W: feature width
    
    Returns:
        distance_bias: tensor of shape (B, num_heads, N, N)
    """
    device = x.device
    # x = x.view(-1, H*W, x.size(-1))
    # B, N, C = x.shape
    num_cav = affine_matrix.size(0)
    
    # 生成坐标网格
    h_idx = torch.arange(H, device=device)
    w_idx = torch.arange(W, device=device)
    grid_y, grid_x = torch.meshgrid(h_idx, w_idx, indexing='ij')
    token_coords = torch.stack([
        grid_x.reshape(-1),
        grid_y.reshape(-1)
    ], dim=-1).float()  # (H*W, 2)
    
    # 计算ego位置
    img_center = torch.tensor([[W/2], [H/2]], device=device).float()
    img_center = img_center.unsqueeze(0).expand(num_cav, -1, -1)
    affine_matrix = affine_matrix.float()
    ego_pos = torch.bmm(affine_matrix[:, :2, :2], img_center) + affine_matrix[:, :2, 2:3]
    ego_pos = ego_pos.squeeze(-1)  # (B, 2)
    
    # 计算相对位置和距离
    token_coords = token_coords.unsqueeze(0)  # (1, N, 2)
    ego_pos = ego_pos.unsqueeze(1)  # (B, 1, 2)
    rel_pos = token_coords - ego_pos  # (B, N, 2)
    
    # 计算距离
    distances = torch.norm(rel_pos, dim=-1)  # (B, N)
    
    # 归一化距离到bins
    dist_normalized = distances / (distances.max() + 1e-6)
    dist_bins = (dist_normalized * (bins-1)).long()  # (B, N)
    
    # 获取距离attention bias (1-2范围)
    distance_bias = tabels[dist_bins]  # (B, N)
    
    # reshape为attention格式
    distance_bias = distance_bias.view(-1, H, W).contiguous().unsqueeze(-1)  # (B, 1, N)

    return distance_bias

def visualize_angle_bias(bias, save_path=None):
    '''
    visualize angle bias
    Args:
        angle_bias: tensor of shape (B, H, W, 1)
        save_path: path to save the visualization
    '''
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Convert to numpy
    bias = bias.squeeze(-1).cpu().detach().numpy()  # (B, H, W)
    bias = bias.reshape(-1, bias.shape[1], bias.shape[2])
    
    #plot
    fig, axes = plt.subplots(1, bias.shape[0], figsize=(15, 5))
    for i in range(bias.shape[0]):
        sns.heatmap(bias[i], ax=axes[i], cmap='RdBu', center=0)
        axes[i].set_title(f'Angle Bias {i+1}')
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()