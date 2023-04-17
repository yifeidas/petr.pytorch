import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_x, pos_y, pos_z), dim=-1)
    return posemb


class FFN(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0., activation='relu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0., activation="relu", n_heads=8):
        super().__init__()
        # pos3d attention
        self.pos3d_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout0 = nn.Dropout(dropout)
        self.norm0 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.ffn = FFN(d_model, d_ffn, dropout, activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, srcs, query_pos=None, posemb_3d=None, posemb_2d=None):
        tgt0 = self.pos3d_attn(self.with_pos_embed(tgt, query_pos).transpose(0, 1),
                                self.with_pos_embed(posemb_3d, posemb_2d).transpose(0,1),
                                posemb_3d.transpose(0, 1))[0].transpose(0,1)
        tgt = tgt + self.dropout0(tgt0)
        tgt = self.norm0(tgt)
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos).transpose(0, 1),
                                self.with_pos_embed(srcs, posemb_3d).transpose(0,1),
                                srcs.transpose(0, 1))[0].transpose(0,1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # ffn
        tgt = self.ffn(tgt)
        return tgt

class Transformer(nn.Module):
    def __init__(self,  d_model=256, nhead=8,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", num_query_position=900, spatial_prior="learned"):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_decoder_layers
        self.num_position = num_query_position
        self.spatial_prior = spatial_prior
        if self.spatial_prior == "learned":
            self.position = nn.Embedding(self.num_position, 3)
            nn.init.uniform_(self.position.weight.data, 0, 1)

        self.position_encoder = nn.Sequential(
            nn.Linear(384, d_model), 
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model), 
        )

        decoder_layer = TransformerDecoderLayer(d_ffn=dim_feedforward, dropout=dropout)
        self.decoder_layers = _get_clones(decoder_layer, num_decoder_layers)

    def forward(self, srcs, posemb_3d=None, posemb_2d=None):
        B, L, C = srcs.size()
        if self.spatial_prior == "learned":
            reference_points = self.position.weight.unsqueeze(0).repeat(B, 1, 1)
        else:
            raise ValueError(f'unknown {self.spatial_prior} spatial prior')

        query_pos = self.position_encoder(pos2posemb3d(reference_points)) # BxNxC

        outputs_feats = []
        outputs_refs = []
        output = torch.zeros_like(query_pos)
        for lid, layer in enumerate(self.decoder_layers):
            output = layer(output, srcs, query_pos=query_pos, posemb_3d=posemb_3d, posemb_2d=posemb_2d)
            output = torch.nan_to_num(output)
            outputs_feats.append(output)
            outputs_refs.append(reference_points.clone())
        return outputs_feats, outputs_refs