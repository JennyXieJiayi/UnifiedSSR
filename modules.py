import math, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, num_layer):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_layer)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask != None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout != None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, emb_size, hid_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(emb_size, hid_size)
        self.w_2 = nn.Linear(hid_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, emb_size, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert emb_size % num_head == 0
        self.d_k = emb_size // num_head
        self.num_head = num_head
        self.linears = clones(nn.Linear(emb_size, emb_size), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask != None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [l(x).view(nbatches, -1, self.num_head, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_head * self.d_k)
        return self.linears[-1](x)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SequencePartition(nn.Module):
    def __init__(self, sub_seq_num, emb_size, is_uniform=False):
        super(SequencePartition, self).__init__()
        self.sub_seq_num = sub_seq_num
        self.offset_activation = nn.ReLU()
        self.proj = nn.AdaptiveAvgPool1d(sub_seq_num)
        self.offset_predictor = nn.Linear(emb_size, 2, bias=False)
        self.sub_seq_coder = SubsequenceCoder(sub_seq_num, is_uniform, width_bias=torch.tensor(5./3.).sqrt().log())

    def forward(self, x):
        src = x
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        x = x.permute(0, 2, 1)
        pred_offset = self.offset_predictor(self.offset_activation(x))
        sub_seq_wins = self.sub_seq_coder(pred_offset)
        sub_seq_wins = sub_seq_wins * src.size(1)
        return sub_seq_wins

    def reset_offset(self):
        nn.init.constant_(self.offset_predictor.weight, 0)
        if hasattr(self.offset_predictor, "bias") and self.offset_predictor.bias is not None:
            nn.init.constant_(self.offset_predictor.bias, 0)


class SubsequenceCoder(nn.Module):
    def __init__(self, sub_seq_num, is_uniform, weights=(1., 1.), width_bias=None):
        super(SubsequenceCoder, self).__init__()
        self.sub_seq_num = sub_seq_num
        self.is_uniform = is_uniform
        self._generate_anchor()
        self.weights = weights  # 2d: center coordinate and length
        self.width_bias = None
        if width_bias is not None:
            self.width_bias = nn.Parameter(width_bias)

    def _generate_anchor(self):
        anchors = []
        sub_seq_stride = 1. / self.sub_seq_num
        for i in range(self.sub_seq_num):
            anchors.append((0.5 + i) * sub_seq_stride)
        anchors = torch.as_tensor(anchors)
        self.register_buffer("anchor", anchors)

    def forward(self, pred_offset):
        if self.is_uniform:
            ref_x = self.anchor.unsqueeze(0)
            windows = torch.zeros_like(pred_offset)
            width = 1 / self.sub_seq_num
            windows[:, :, 0] = ref_x - width / 2
            windows[:, :, 1] = ref_x + width / 2
        else:
            if self.width_bias is not None:
                pred_offset[:, :, -1] = pred_offset[:, :, -1] + self.width_bias
            windows = self.decode(pred_offset)

        windows = windows.clamp(min=0., max=1.)
        return windows

    def decode(self, rel_codes):
        windows = self.anchor
        point = 1. / self.sub_seq_num
        w_x, w_width = self.weights

        dx = torch.tanh(rel_codes[:, :, 0] / w_x) * point
        dw = F.relu(torch.tanh(rel_codes[:, :, -1] / w_width)) * point

        pred_windows = torch.zeros_like(rel_codes)
        ref_x = windows.unsqueeze(0)
        pred_windows[:, :, 0] = ref_x + dx - dw
        pred_windows[:, :, -1] = ref_x + dx + dw
        pred_windows = pred_windows.clamp(min=0., max=1.)
        return pred_windows


class SiameseEncoder(nn.Module):
    def __init__(self, layer, num_layer):
        super(SiameseEncoder, self).__init__()
        self.layers = clones(layer, num_layer)
        self.norm = LayerNorm(layer.emb_size)

    def forward(self, src1, src2, mask):
        for layer in self.layers:
            x = layer(src1, src2, mask)
        return self.norm(x)


class SiameseEncoderLayer(nn.Module):
    def __init__(self, emb_size, hid_size, num_head, dropout):
        super(SiameseEncoderLayer, self).__init__()
        self.emb_size = emb_size
        self.self_attn = MultiHeadAttention(num_head, emb_size)
        self.cross_attn = MultiHeadAttention(num_head, emb_size)
        self.feed_forward = PositionwiseFeedForward(emb_size, hid_size, dropout)
        self.sublayer = clones(SublayerConnection(emb_size, dropout), 3)

    def forward(self, src1, src2, src_mask):
        src1 = self.sublayer[0](src1, lambda src1: self.self_attn(src1, src1, src1, src_mask))
        src1 = self.sublayer[1](src1, lambda src1: self.cross_attn(src1, src2, src2, src_mask))
        return self.sublayer[2](src1, self.feed_forward)


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., emb_size, 2) *
                             -(math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(-2)]
        return self.dropout(x)


class Embeddings(nn.Module):
    def __init__(self, vocab, emb_size):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, emb_size)
        self.emb_size = emb_size

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.emb_size)