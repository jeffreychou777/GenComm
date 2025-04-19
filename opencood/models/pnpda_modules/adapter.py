import math
from einops import einsum, rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.utils.pe import PositionEmbeddingLearned, PositionEmbeddingSine
from opencood.models.sub_modules.base_transformer import FeedForward, PreNormResidual


class proj_k(nn.Module):
    def __init__(self, args) -> None:
        super(proj_k, self).__init__()
        self.proj_k = TransformerEncoder(args)

    def forward(self, x):
        return self.proj_k(x)
    

class proj_q(nn.Module):
    def __init__(self, args) -> None:
        super(proj_q, self).__init__()
        self.proj_q = TransformerEncoder(args)

    def forward(self, x):
        return self.proj_q(x)
    
class predictor(nn.Module):
    def __init__(self, args) -> None:
        super(predictor, self).__init__()
        self.projector = TransformerDecoder(args)

    def forward(self, query, key):
        return self.projector(query, key)


class CrossAttention(nn.Module):
    def __init__(self, d_model, heads, d_ff, qkv_bias):
        super().__init__()

        dim_head = d_ff // heads
        self.scale = dim_head**-0.5

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_ff, bias=qkv_bias)
        )
        self.to_k = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_ff, bias=qkv_bias)
        )
        self.to_v = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_ff, bias=qkv_bias)
        )

        self.proj = nn.Linear(d_ff, d_model)

    def forward(self, q, k, v, skip=None):
        """
        q: (b X Y W1 W2 d)
        k: (b x y w1 w2 d)
        v: (b x y w1 w2 d)
        return: (b X Y W1 W2 d)
        """
        assert k.shape == v.shape
        _, q_height, q_width, q_win_height, q_win_width, _ = q.shape
        _, kv_height, kv_width, _, _, _ = k.shape
        assert q_height * q_width == kv_height * kv_width
        # print(q.shape, k.shape)

        # flattening
        q = rearrange(q, "b x y w1 w2 d -> b (x y) (w1 w2) d")
        k = rearrange(k, "b x y w1 w2 d -> b (x y) (w1 w2) d")
        v = rearrange(v, "b x y w1 w2 d -> b (x y) (w1 w2) d")

        # Project with multiple heads
        q = self.to_q(q)  # b (X Y) (W1 W2) (heads dim_head)
        k = self.to_k(k)  # b (X Y) (w1 w2) (heads dim_head)
        v = self.to_v(v)  # b (X Y) (w1 w2) (heads dim_head)
        # print(q.shape,k.shape,v.shape)

        # Group the head dim with batch dim
        q = rearrange(q, "b ... (m d) -> (b m) ... d", m=self.heads, d=self.dim_head)
        k = rearrange(k, "b ... (m d) -> (b m) ... d", m=self.heads, d=self.dim_head)
        v = rearrange(v, "b ... (m d) -> (b m) ... d", m=self.heads, d=self.dim_head)
        # print(q.shape,k.shape,v.shape)

        # cross attention between cav and ego feature
        dot = self.scale * torch.einsum(
            "b l Q d, b l K d -> b l Q K", q, k
        )  # b (X Y) (W1 W2) (w1 w2)
        att = dot.softmax(dim=-1)

        a = torch.einsum("b n Q K, b n K d -> b n Q d", att, v)  # b (X Y) (W1 W2) d
        # print('a',a.shape)
        a = rearrange(a, "(b m) ... d -> b ... (m d)", m=self.heads, d=self.dim_head)
        # print('a',a.shape)
        a = rearrange(
            a,
            " b (x y) (w1 w2) d -> b x y w1 w2 d",
            x=q_height,
            y=q_width,
            w1=q_win_height,
            w2=q_win_width,
        )
        # Combine multiple heads
        z = self.proj(a)

        # Optional skip connection
        if skip is not None:
            z = z + skip

        return z


class SelfAttention(nn.Module):
    def __init__(self, d_model, heads, d_ff, drop_out):
        super().__init__()

        dim_head = d_ff // heads
        self.scale = dim_head**-0.5

        self.heads = heads
        self.dim_head = dim_head
        
        self.to_q = nn.Linear(d_model, d_ff)
        self.to_k = nn.Linear(d_model, d_ff)
        self.to_v = nn.Linear(d_model, d_ff)
        
        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(drop_out))
        self.to_out = nn.Sequential(
            nn.Linear(d_ff, d_model, bias=False), nn.Dropout(drop_out)
        )


    def forward(self, q,k,v):
        """
        q: (b h w d)
        k: (b h w d)
        v: (b h w d)
        return: (b h w d)
        """
        assert k.shape == v.shape
        _, q_height, q_width, _ = q.shape
        _, kv_height, kv_width, _ = k.shape
        assert q_height * q_width == kv_height * kv_width

        
        # Project with multiple heads
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        # Group the head dim with batch dim
        q = rearrange(q, "b ... (m d) -> (b m) ... d", m=self.heads, d=self.dim_head)
        k = rearrange(k, "b ... (m d) -> (b m) ... d", m=self.heads, d=self.dim_head)
        v = rearrange(v, "b ... (m d) -> (b m) ... d", m=self.heads, d=self.dim_head)

        # cross attention between cav and ego feature
        dot = self.scale * torch.einsum("b l Q d, b l K d -> b l Q K", q, k)
        att = self.attend(dot)

        a = torch.einsum("b n Q K, b n K d -> b n Q d", att, v)  # (b*head h w d/head)
        a = rearrange(a, "(b m) ... d -> b ... (m d)", m=self.heads, d=self.dim_head)
        # print('a',a.shape)

        return self.to_out(a)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(DecoderLayer, self).__init__()
        self.win_size = 2

        self.prenorm1 = nn.LayerNorm(d_model)
        self.prenorm2 = nn.LayerNorm(d_model)
        self.mlp_1 = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model)
        )
        self.mlp_2 = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model)
        )

        self.cross_win_1 = CrossAttention(d_model, num_heads, d_ff, qkv_bias=True)
        self.cross_win_2 = CrossAttention(d_model, num_heads, d_ff, qkv_bias=True)

        self.win_size = 2

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, ego, cav_feature, pos=None):
        """
        Parameters
        ----------
        ego : b h w c
        cav_feature : b h w c
        """
        query = ego
        key = cav_feature
        value = cav_feature
        # print('query',query.shape)
        # print('key',key.shape)

        # local attention
        query = rearrange(
            query,
            "b (x w1) (y w2) d  -> b x y w1 w2 d",
            w1=self.win_size,
            w2=self.win_size,
        )  # window partition
        key = rearrange(
            key,
            "b (x w1) (y w2) d  -> b x y w1 w2 d",
            w1=self.win_size,
            w2=self.win_size,
        )  # window partition
        value = rearrange(
            value,
            "b (x w1) (y w2) d -> b x y w1 w2 d",
            w1=self.win_size,
            w2=self.win_size,
        )  # window partition
        pos = rearrange(
            pos,
            "b (x w1) (y w2) d -> b x y w1 w2 d",
            w1=self.win_size,
            w2=self.win_size,
        )  # window partition

        key = rearrange(
            self.cross_win_1(
                self.with_pos_embed(query, pos),
                self.with_pos_embed(key, pos),
                value,
                skip=query,
            ),
            "b x y w1 w2 d  -> b (x w1) (y w2) d",
        )
        key = self.prenorm1(key + self.mlp_1(key))
        query = rearrange(query, "b x y w1 w2 d  -> b (x w1) (y w2) d")
        value = rearrange(value, "b x y w1 w2 d  -> b (x w1) (y w2) d")
        pos = rearrange(pos, "b x y w1 w2 d  -> b (x w1) (y w2) d")

        # global attention
        query = rearrange(
            query,
            "b (w1 x) (w2 y) d -> b x y w1 w2 d",
            w1=self.win_size,
            w2=self.win_size,
        )
        key = rearrange(
            key,
            "b (w1 x) (w2 y) d -> b x y w1 w2 d",
            w1=self.win_size,
            w2=self.win_size,
        )
        value = rearrange(
            value,
            "b (w1 x) (w2 y) d -> b x y w1 w2 d",
            w1=self.win_size,
            w2=self.win_size,
        )
        pos = rearrange(
            pos,
            "b (w1 x) (w2 y) d -> b x y w1 w2 d",
            w1=self.win_size,
            w2=self.win_size,
        )
        key = rearrange(
            self.cross_win_2(
                self.with_pos_embed(query, pos),
                self.with_pos_embed(key, pos),
                value,
                skip=query,
            ),
            "b x y w1 w2 d  -> b (w1 x) (w2 y) d",
        )
        key = self.prenorm2(key + self.mlp_2(key))

        key = rearrange(key, "b h w d -> b d h w")

        return key


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()

        self.self_attention = SelfAttention(d_model, num_heads, d_ff, dropout)
        self.feedforward = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, pos):
        q = k = self.with_pos_embed(x, pos)
        _x = self.self_attention(q, k, x)
        x = self.norm1(_x + x)
        _x = self.feedforward(x)
        x = self.norm2(x + _x)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, args) -> None:
        super(TransformerDecoder, self).__init__()
        self.d_model = args["d_model"]
        self.num_heads = args["num_heads"]
        self.d_ff = args["d_ff"]
        self.num_layers = args["num_layers"]
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_ff=self.d_ff,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.pos_embed_sin = PositionEmbeddingSine(d_model=self.d_model)

    def forward(self, ego_feature, cav_feature):
        # b,h,w,c
        ego_feature = ego_feature.permute(0, 2, 3, 1).contiguous()
        cav_feature = cav_feature.permute(0, 2, 3, 1).contiguous()
        pos = self.pos_embed_sin(ego_feature)
        output = cav_feature

        for i, layer in enumerate(self.layers):
            output = layer(ego_feature, output, pos)

        return output


class TransformerEncoder(nn.Module):
    def __init__(self, args) -> None:
        super(TransformerEncoder, self).__init__()
        self.d_model = args["d_model"]
        self.num_heads = args["num_heads"]
        self.d_ff = args["d_ff"]
        self.num_layers = args["num_layers"]
        self.dropout = args["dropout"]
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_ff=self.d_ff,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.pos_embed_sin = PositionEmbeddingSine(d_model=self.d_model)

    def forward(self, x):
        # b,c,h,w
        x = x.permute(0, 2, 3, 1).contiguous()
        pos = self.pos_embed_sin(x)
        # print(x.shape,pos.shape)

        for i, layer in enumerate(self.layers):
            x = layer(x, pos)

        return x.permute(0, 3, 1, 2)


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ego = torch.rand(1, 50, 176, 256)  # .cuda()
    cav = torch.rand(1, 50, 176, 256)  # .cuda()
