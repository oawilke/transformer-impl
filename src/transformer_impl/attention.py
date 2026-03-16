import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionHead(nn.Module):
    def __init__(self, d_embedding, d_k, d_v, bias):
        super().__init__()
        self.d_embedding = d_embedding
        self.d_k = d_k
        self.d_v = d_v

        self.WQ = nn.Linear(d_embedding, d_k, bias=bias)
        self.WK = nn.Linear(d_embedding, d_k, bias=bias)
        self.WV = nn.Linear(d_embedding, d_v, bias=bias)

    def forward(self, x_encoder, x_decoder, mask = None):
        q = self.WQ(x_decoder)
        k = self.WK(x_encoder)
        v = self.WV(x_encoder)

        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.to(torch.bool)

            attention_scores = attention_scores.masked_fill(~mask, float('-inf'))

        attention_probs = F.softmax(attention_scores, dim = -1)

        return attention_probs @ v


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, n_heads, d_embedding, bias):
        super().__init__()

        d_head = d_embedding // n_heads
        assert d_embedding % n_heads == 0

        self.n_heads = n_heads
        self.heads = nn.ModuleList([CrossAttentionHead(d_embedding, d_head, d_head, bias) for _ in range(n_heads)])
        self.WO = nn.Linear(d_embedding, d_embedding, bias=bias)

    def forward(self, x_encoder, x_decoder, mask = None):
        head_outputs = [head(x_encoder, x_decoder, mask) for head in self.heads]
        concatenated_head_outputs = torch.cat(head_outputs, dim = -1)
        return self.WO(concatenated_head_outputs)


class SelfAttentionHead(nn.Module):
    def __init__(self, d_embedding, d_k, d_v, bias):
        super().__init__()
        self.d_embedding = d_embedding
        self.d_k = d_k
        self.d_v = d_v

        self.WQ = nn.Linear(d_embedding, d_k, bias=bias)
        self.WK = nn.Linear(d_embedding, d_k, bias=bias)
        self.WV = nn.Linear(d_embedding, d_v, bias=bias)

    def forward(self, x, mask = None):
        q = self.WQ(x)
        k = self.WK(x)
        v = self.WV(x)

        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.to(torch.bool)

            attention_scores = attention_scores.masked_fill(~mask, float('-inf'))

        attention_probs = F.softmax(attention_scores, dim = -1)
    
        return attention_probs @ v


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_heads, d_embedding, bias):
        super().__init__()

        d_head = d_embedding // n_heads
        assert d_embedding % n_heads == 0

        self.n_heads = n_heads
        self.heads = nn.ModuleList([SelfAttentionHead(d_embedding, d_head, d_head, bias) for _ in range(n_heads)])
        self.WO = nn.Linear(d_embedding, d_embedding, bias=bias)

    def forward(self, x, mask = None):
        head_outputs = [head(x, mask) for head in self.heads]
        concatenated_head_outputs = torch.cat(head_outputs, dim = -1)
        return self.WO(concatenated_head_outputs)
