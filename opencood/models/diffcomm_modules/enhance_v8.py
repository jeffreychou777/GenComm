import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np
import time
from torch import einsum
from opencood.models.fuse_modules.fusion_in_one import regroup

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)
        


#########################################
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1):
        super(ConvBlock, self).__init__()
        self.strides = strides
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.conv11 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0)

    def forward(self, x):
        out1 = self.block(x)
        out2 = self.conv11(x)
        out = out1 + out2
        return out

class LinearProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.repeat(B_,1,1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1] 
        return q,k,v

class LinearProjection_cross(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_,1,1)
        else:
            attn_kv = x
        q = x[0].unsqueeze(0)
        k_v = x[1:]
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1] 
        return q,k,v


#########################################
########### window-based self-attention #############
class WindowAttention(nn.Module):
    def __init__(self, dim, win_size,num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0]) # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1]) # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)
            
        if token_projection =='linear':
            self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        else:
            raise Exception("Projection error!") 
        
        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x,attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1)//relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d = ratio)
    
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)',d = ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N*ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N*ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'

########### window-based self-attention #############
class WindowAttention_sparse(nn.Module):
    def __init__(self, dim, win_size,num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        
        # 新增: ego-relative位置编码
        self.ego_dist_bins = 4  # 距离bins数量
        self.ego_angle_bins = 4  # 角度bins数量 
        
        # ego相对位置编码表
        self.ego_position_bias_table = nn.Parameter(
            torch.zeros(self.ego_dist_bins * self.ego_angle_bins, num_heads))

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0]) # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1]) # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)
            
        if token_projection =='linear':
            self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        elif token_projection =='cross':
            self.qkv = LinearProjection_cross(dim,num_heads,dim//num_heads,bias=qkv_bias)
        else:
            raise Exception("Projection error!") 
        
        
        
        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.w = nn.Parameter(torch.ones(2)) 
    
    def get_ego_relative_bias(self, x_windows, affine_matrix, H=None, W=None):
        """
        计算每个CAV的每个window相对于ego vehicle的位置编码
        
        Args:
            x_windows: tensor of shape (num_cav*num_windows, win_size*win_size, C)
            affine_matrix: tensor of shape (num_cav, 2, 3)
        
        Returns:
            ego_relative_bias: tensor of shape (num_cav*num_windows, num_heads, win_size*win_size, win_size*win_size)
        """
        device = x_windows.device
        num_cav = affine_matrix.size(0)
        num_windows_per_cav = x_windows.size(0) // num_cav
        win_h = win_w = self.win_size[0]

        
        # 为每个CAV生成window中心坐标
        h_idx = torch.arange(0, H//win_h, device=device)
        w_idx = torch.arange(0, W//win_w, device=device)
        grid_y, grid_x = torch.meshgrid(h_idx, w_idx, indexing='ij')
        window_centers = torch.stack([
            (grid_x * win_w + win_w/2).reshape(-1),  # x coordinates
            (grid_y * win_h + win_h/2).reshape(-1)   # y coordinates
        ], dim=-1).float()  # (num_windows_per_cav, 2)
        
        # 计算每个CAV的ego位置
        img_center = torch.tensor([[W/2], [H/2]], device=device).float()
        img_center = img_center.unsqueeze(0).expand(num_cav, -1, -1)  # (num_cav, 2, 1)
        affine_matrix = affine_matrix.float()
        ego_pos = torch.bmm(affine_matrix[:, :2, :2], img_center) + affine_matrix[:, :2, 2:3]  # (num_cav, 2, 1)
        ego_pos = ego_pos.squeeze(-1)  # (num_cav, 2)
        
        # 为每个CAV计算所有windows的相对位置
        window_centers = window_centers.unsqueeze(0)  # (1, num_windows_per_cav, 2)
        ego_pos = ego_pos.unsqueeze(1)  # (num_cav, 1, 2)
        rel_pos = window_centers - ego_pos  # (num_cav, num_windows_per_cav, 2)
        
        # 计算距离和角度
        distances = torch.norm(rel_pos, dim=-1)  # (num_cav, num_windows_per_cav)
        angles = torch.atan2(rel_pos[..., 1], rel_pos[..., 0])  # (num_cav, num_windows_per_cav)
        
        # 归一化到bins
        dist_normalized = distances / (distances.max() + 1e-6)
        dist_bins = (dist_normalized * (self.ego_dist_bins-1)).long()  # (num_cav, num_windows_per_cav)

        angle_normalized = (angles + math.pi) / (2 * math.pi)
        angle_bins = (angle_normalized * (self.ego_angle_bins-1)).long()  # (num_cav, num_windows_per_cav)
        
        # 计算position index
        position_index = dist_bins * self.ego_angle_bins + angle_bins  # (num_cav, num_windows_per_cav)
        
        # 将所有CAV的windows展平
        position_index = position_index.reshape(-1)  # (num_cav*num_windows_per_cav)
        
        # 获取position embeddings
        ego_relative_bias = self.ego_position_bias_table[position_index]  # (num_cav*num_windows_per_cav, num_heads)
        
        # reshape为attention bias格式
        ego_relative_bias = ego_relative_bias.unsqueeze(1).unsqueeze(2)  # (num_cav*num_windows_per_cav, 1, 1, num_heads)
        ego_relative_bias = ego_relative_bias.expand(
            -1, win_h*win_w, win_h*win_w, -1)  # (num_cav*num_windows_per_cav, win_size^2, win_size^2, num_heads)
        ego_relative_bias = ego_relative_bias.permute(0, 3, 1, 2)  # (num_cav*num_windows_per_cav, num_heads, win_size^2, win_size^2)
        
        return ego_relative_bias
    
    def visualize_ego_relative_bias(self, save_path=None):
        """
        Visualize the ego-relative position bias table as heatmaps for each attention head.
        
        Args:
            save_path (str, optional): Path to save the visualization. If None, display plot.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Get bias table
        bias_table = self.ego_position_bias_table.detach().cpu().numpy()  # [dist_bins*angle_bins, num_heads]
        dist_bins = self.ego_dist_bins
        angle_bins = self.ego_angle_bins
        num_heads = bias_table.shape[1]

        # Reshape bias table to [dist_bins, angle_bins, num_heads]
        bias_map = bias_table.reshape(dist_bins, angle_bins, num_heads)

        # Create figure with subplots for each head
        fig, axes = plt.subplots(2, (num_heads+1)//2, figsize=(4*num_heads, 8))
        axes = axes.flatten()

        # Create angle labels
        angle_labels = [f'{int(360*i/angle_bins)}°' for i in range(angle_bins)]
        
        # Create distance labels
        dist_labels = [f'D{i+1}' for i in range(dist_bins)]

        # Plot heatmap for each attention head
        for head in range(num_heads):
            sns.heatmap(bias_map[:, :, head],
                    ax=axes[head],
                    cmap='RdBu',
                    center=0,
                    xticklabels=angle_labels,
                    yticklabels=dist_labels,
                    cbar_kws={'label': 'Bias Value'})
            
            axes[head].set_title(f'Head {head+1}')
            axes[head].set_xlabel('Angle')
            axes[head].set_ylabel('Distance')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
        
    def forward(self, x, attn_kv=None, mask=None, affine_matrix = None, H=None, W=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x,attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1)//relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d = ratio)

        if affine_matrix is not None:
            ego_relative_bias = self.get_ego_relative_bias(x, affine_matrix, H, W)
            attn = attn + relative_position_bias.unsqueeze(0) + ego_relative_bias
        else:
            attn = attn + relative_position_bias.unsqueeze(0)
        
        
        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)',d = ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N*ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N*ratio)
            attn0 = self.softmax(attn)
            # attn1 = self.relu(attn)**2
            attn1 = self.relu(attn)#b,h,w,c
        else:
            attn0 = self.softmax(attn)
            attn1 = self.relu(attn)**2
            # attn1 = self.relu(attn)
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        attn = attn0*w1+attn1*w2
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        self.visualize_ego_relative_bias('/home/junfei.zhou/DATACENTER2/data/code/DiffComm/bias_vis')
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'

########### self-attention #############
class Attention(nn.Module):
    def __init__(self, dim, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        # 角度相关的参数
        self.angle_bins = 5  # 角度bins数量
        # 初始化为1，让bias的范围在1-2之间
        self.angle_bias_table = nn.Parameter(torch.ones(self.angle_bins, num_heads))
            
        self.qkv = LinearProjection(dim, num_heads, dim//num_heads, bias=qkv_bias)
        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def get_angle_attention_bias(self, x, affine_matrix, H, W):
        """
        计算基于角度的attention bias (范围1-2)
        """
        device = x.device
        B, N, C = x.shape
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
        angle_bins = (angle_normalized * (self.angle_bins-1)).long()  # (B, N)
        
        # 获取角度attention bias (1-2范围)
        angle_bias = 1.0 + F.sigmoid(self.angle_bias_table[angle_bins])  # (B, N, num_heads)
        
        # reshape为attention格式
        angle_bias = angle_bias.permute(0, 2, 1).unsqueeze(-1)  # (B, num_heads, N, 1)
        angle_bias = angle_bias.expand(-1, -1, -1, N)  # (B, num_heads, N, N)
        
        return angle_bias

    def forward(self, x, attn_kv=None, mask=None, affine_matrix=None, H=None, W=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # 添加角度attention bias
        if affine_matrix is not None:
            angle_bias = self.get_angle_attention_bias(x, affine_matrix, H, W)
            attn = attn * angle_bias  # 乘法形式应用bias

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'



#########################################
########### feed-forward network #############
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU,drop = 0., use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim,hidden_dim,groups=hidden_dim,kernel_size=3,stride=1,padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.eca = nn.Identity()

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h = hh, w = hh)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h = hh, w = hh)

        x = self.linear2(x)
        x = self.eca(x)

        return x


class FRFN(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU,drop = 0., use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim*2),
                                act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim,hidden_dim,groups=hidden_dim,kernel_size=3,stride=1,padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.dim_conv = self.dim // 4
        self.dim_untouched = self.dim - self.dim_conv 
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)

    def forward(self, x, H, W):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))


        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h = H, w = W)

        x1, x2,= torch.split(x, [self.dim_conv,self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h = H, w = W)

        x = self.linear1(x)
        #gate mechanism
        x_1,x_2 = x.chunk(2,dim=-1)

        x_1 = rearrange(x_1, ' b (h w) (c) -> b c h w ', h = H, w = W)
        x_1 = self.dwconv(x_1)
        x_1 = rearrange(x_1, ' b c h w -> b (h w) c', h = H, w = W)
        x = x_1 * x_2
        
        x = self.linear2(x)
        # x = self.eca(x)

        return x



#########################################
########### window operation#############
def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate !=1:
        x = x.permute(0,3,1,2) # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size,dilation=dilation_rate,padding=4*(dilation_rate-1),stride=win_size) # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0,2,1).contiguous().view(-1, C, win_size, win_size) # B' ,C ,Wh ,Ww
        windows = windows.permute(0,2,3,1).contiguous() # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C) # B' ,Wh ,Ww ,C
    return windows

def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate !=1:
        x = windows.permute(0,5,3,4,1,2).contiguous() # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4*(dilation_rate-1),stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class RadixSoftmax(nn.Module):
    def __init__(self, radix, cardinality):
        super(RadixSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        # x: (B, L, 1, 1, 3C)
        cav_num = x.size(0)

        if self.radix > 1:
            # x: (B, L, 1, 3, C)
            x = x.view(cav_num,
                       self.cardinality, self.radix, -1)
            x = F.softmax(x, dim=2)
            # B, 3LC
            x = x.reshape(cav_num, -1)
        else:
            x = torch.sigmoid(x)
        return x

class SplitAttn(nn.Module):
    def __init__(self, input_dim):
        super(SplitAttn, self).__init__()
        self.input_dim = input_dim

        self.fc1 = nn.Linear(input_dim, input_dim, bias=False)
        self.bn1 = nn.LayerNorm(input_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(input_dim, input_dim * 3, bias=False)

        self.rsoftmax = RadixSoftmax(3, 1)

    def forward(self, window_list):
        # window list: [(B, L, H, W, C) * 3]
        assert len(window_list) == 3, 'only 3 windows are supported'

        sw, mw, bw = window_list[0], window_list[1], window_list[2]
        cav_nums = sw.shape[0],

        # global average pooling, B, L, H, W, C
        x_gap = sw + mw + bw
        # B, L, 1, 1, C
        x_gap = x_gap.mean((1, 2), keepdim=True)
        x_gap = self.act1(self.bn1(self.fc1(x_gap)))
        # B, L, 1, 1, 3C
        x_attn = self.fc2(x_gap)
        # B L 1 1 3C
        x_attn = self.rsoftmax(x_attn).unsqueeze(1).unsqueeze(1)

        out = sw * x_attn[:, :, :, 0:self.input_dim] + \
              mw * x_attn[:, :, :, self.input_dim:2*self.input_dim] +\
              bw * x_attn[:, :, :, self.input_dim*2:]

        return out

class Enhancer_block(nn.Module):
    def __init__(self, C, win_size, num_heads,):
        super().__init__()
        self.window_size = win_size
        self.attn = Attention(dim=C, num_heads=num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.1, proj_drop=0.1, token_projection='linear')
        self.mlp = FRFN(dim=C, hidden_dim=C*2, act_layer=nn.GELU, drop=0.)
        self.norm1 = nn.LayerNorm(C)
        self.norm2 = nn.LayerNorm(C)        
        self.drop_path = DropPath(0.1) if 0.1 > 0. else nn.Identity()
        self.drop_path = nn.Identity()
        
    def forward(self, x, affine_matrix=None):
        B, C, H, W = x.shape
        x = x.permute(0,2,3,1).contiguous()
        x = x.view(B, H*W, C)
        shortcut = x
        x = self.norm1(x)        
        # x = self.attn(x, affine_matrix=affine_matrix, H=H, W=W)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        x = x.view(B, H, W, C)

        return x

class Enhancerv8(nn.Module):
    '''
    v8 is single one 
    '''
    def __init__(self, C, win_size, num_heads):
        super().__init__()
        self.block_1 = Enhancer_block(C, [4,4], num_heads)
        # self.block_2 = Enhancer_block(C, win_size, num_heads)
        # self.block_3 = Enhancer_block(C, [16, 16], num_heads)
        # self.split_attn = SplitAttn(C)

    def forward(self, x, affine_matrix=None, record_len=None):
        cav_num, C, H, W = x.shape
        split_x = regroup(x, record_len)
        out = []
        for b in range(len(record_len)):
            x = split_x[b]
            # if x.shape[0] == 1:
            #     out.append(x)
            affine_matrix_ = affine_matrix[b,0][:x.shape[0]]
            s = self.block_1(x, affine_matrix_)
            # m = self.block_2(x, affine_matrix_)
            # l = self.block_3(x, affine_matrix_)
            # out.append(self.split_attn([s, m, l]).permute(0,3,1,2).contiguous())
            out.append(s.permute(0,3,1,2).contiguous())
            
        out = torch.cat(out, 0)
        return out

          
  
#########################################
# window attention sparse test #
def count_parameters(model):
    """计算模型的总参数量（可训练 + 不可训练）。"""
    total_params = sum(p.numel() for p in model.parameters())  # 计算所有参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 仅计算可训练参数
    return {'Total': total_params, 'Trainable': trainable_params}


# def test_sparse():
#     affine_matrix = torch.randn(1,5,5,2,3)
#     B = 2
#     H = 128
#     W = 256
#     C = 128
#     N = H*W
#     ratio = 4
#     win_size = [8,8]
#     num_heads = 4
#     x = torch.randn(B, C, H, W)
#     block = Enhancer(C, win_size, num_heads)
#     y = block(x, affine_matrix)

#     print(y.shape)
#     print(count_parameters(block))
    
    

# test_sparse()
