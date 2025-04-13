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
        self.fc2 = nn.Linear(input_dim, input_dim, bias=False)

        self.rsoftmax = RadixSoftmax(1, 1)
        # self.softmax = nn.Softmax(dim=-1)
        
        # 角度相关的参数
        self.angle_bins = 5  # 角度bins数量
        # 初始化为1，让bias的范围在1-2之间
        self.angle_bias_table = nn.Parameter(1 + torch.rand(self.angle_bins))
        
        self.distance_bins = 5  # 距离bins数量
        self.distance_bias_table = nn.Parameter(1 + torch.rand(self.angle_bins))
    
    def get_angle_bias(self, x, affine_matrix, H, W):
        """
        计算基于角度的attention bias (范围1-2)
        """
        device = x.device
        x = x.view(-1, H*W, x.size(-1))  # (B, H, W, C)
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
        angle_bias = torch.clamp(self.angle_bias_table[angle_bins], min=1.0, max=2.0) # (B, N, num_heads)
        
        # reshape为attention格式
        angle_bias = angle_bias.view(-1, H, W).contiguous().unsqueeze(-1)  # (B, N, 1)
        
        return angle_bias
    
    def get_distance_bias(self, x, affine_matrix, H, W):
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
        x = x.view(-1, H*W, x.size(-1))
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
        
        # 计算相对位置和距离
        token_coords = token_coords.unsqueeze(0)  # (1, N, 2)
        ego_pos = ego_pos.unsqueeze(1)  # (B, 1, 2)
        rel_pos = token_coords - ego_pos  # (B, N, 2)
        
        # 计算距离
        distances = torch.norm(rel_pos, dim=-1)  # (B, N)
        
        # 归一化距离到bins
        dist_normalized = distances / (distances.max() + 1e-6)
        dist_bins = (dist_normalized * (self.distance_bins-1)).long()  # (B, N)
        
        # 获取距离attention bias (1-2范围)
        distance_bias = torch.clamp(self.distance_bias_table[dist_bins], min=1.0, max=2.0)  # (B, N)
        
        # reshape为attention格式
        distance_bias = distance_bias.view(-1, H, W).contiguous().unsqueeze(-1)  # (B, 1, N)

        return distance_bias
        
    def visualize_angle_bias(self, angle_bias, save_path=None):
        '''
        visualize angle bias
        Args:
            angle_bias: tensor of shape (B, H, W, 1)
            save_path: path to save the visualization
        '''
        import matplotlib.pyplot as plt
        import seaborn as sns
        # Convert to numpy
        angle_bias = angle_bias.squeeze(-1).cpu().detach().numpy()  # (B, H, W)
        angle_bias = angle_bias.reshape(-1, angle_bias.shape[1], angle_bias.shape[2])
        
        #plot
        fig, axes = plt.subplots(1, angle_bias.shape[0], figsize=(15, 5))
        for i in range(angle_bias.shape[0]):
            sns.heatmap(angle_bias[i], ax=axes[i], cmap='RdBu', center=0)
            axes[i].set_title(f'Angle Bias {i+1}')
            axes[i].set_xlabel('X')
            axes[i].set_ylabel('Y')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()

    def forward(self, window_list,):
        # window list: [(B, L, H, W, C) * 3]
        assert len(window_list) == 1, 'only 3 windows are supported'

        sw = window_list[0]
        cav_nums = sw.shape[0],

        # global average pooling, B, L, H, W, C
        x_gap = sw
        # B, L, 1, 1, C
        x_gap = x_gap.mean((1, 2), keepdim=True)
        x_gap = self.act1(self.bn1(self.fc1(x_gap)))
        # B, L, 1, 1, 3C
        x_attn = self.fc2(x_gap)
        # B L 1 1 3C
        x_attn = self.rsoftmax(x_attn)
        
        out = sw * x_attn[:, :, :, 0:self.input_dim]

        # self.visualize_angle_bias(angle_bias, save_path='/home/junfei.zhou/DATACENTER2/data/code/DiffComm/bias_vis/angle_bias.png')
        # # only angle bias
        # out = sw * x_attn[:, :, :, 0:self.input_dim] + \
        #       mw * x_attn[:, :, :, self.input_dim:2*self.input_dim] * angle_bias +\
        #       bw * x_attn[:, :, :, self.input_dim*2:]
        
        # only dist bias
        # out = sw * x_attn[:, :, :, 0:self.input_dim] + \
        #       mw * x_attn[:, :, :, self.input_dim:2*self.input_dim] * distance_bias +\
        #       bw * x_attn[:, :, :, self.input_dim*2:]
        return out

class Enhancer_block(nn.Module):
    def __init__(self, C,):
        super().__init__()
        self.mlp = FRFN(dim=C, hidden_dim=C*2, act_layer=nn.GELU, drop=0.)
        self.norm1 = nn.LayerNorm(C)
        self.norm2 = nn.LayerNorm(C)        
        self.drop_path = nn.Identity()
        
    def forward(self, x,):
        B, C, H, W = x.shape
        x = x.permute(0,2,3,1).contiguous()
        x = x.view(B, H*W, C)
        # shortcut = x
        # x = self.norm1(x)        
        # x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        x = x.view(B, H, W, C)

        return x

class Enhancer(nn.Module):
    '''
    v12 is v7  single and channel attention
    '''
    def __init__(self, C):
        super().__init__()
        self.block_1 = Enhancer_block(C,)
        self.block_2 = Enhancer_block(C,)
        self.block_3 = Enhancer_block(C,)
        self.split_attn = SplitAttn(C)

    def forward(self, x, affine_matrix=None, record_len=None):
        cav_num, C, H, W = x.shape
        split_x = regroup(x, record_len)
        out = []
        for b in range(len(record_len)):
            x = split_x[b]
            # if x.shape[0] == 1:
            #     out.append(x)
            # affine_matrix_ = affine_matrix[b,0][:x.shape[0]]
            s = self.block_1(x, )
            # m = self.block_2(x, affine_matrix_)
            # l = self.block_3(x, affine_matrix_)
            # out.append(self.split_attn([s, m, l], affine_matrix_).permute(0,3,1,2).contiguous())
            out.append(self.split_attn([s,],).permute(0,3,1,2).contiguous())
            
        out = torch.cat(out, 0)
        return out

          
  
#########################################
# window attention sparse test #
def count_parameters(model):
    """计算模型的总参数量（可训练 + 不可训练）。"""
    total_params = sum(p.numel() for p in model.parameters())  # 计算所有参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 仅计算可训练参数
    return {'Total': total_params, 'Trainable': trainable_params}


def test_sparse():
    affine_matrix = torch.randn(2,5,5,2,3)
    record_len = torch.tensor([2,1])
    B = 3
    H = 128
    W = 256
    C = 128
    N = H*W
    ratio = 4
    win_size = [8,8]
    num_heads = 4
    x = torch.randn(B, C, H, W)
    block = Enhancer(C, win_size, num_heads,)
    y = block(x, affine_matrix, record_len)

    print(y.shape)
    print(count_parameters(block))
    
    

# test_sparse()
