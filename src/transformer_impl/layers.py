import torch.nn as nn

from .attention import MultiHeadCrossAttention, MultiHeadSelfAttention


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_embedding, bias, p_drop):
        super().__init__()

        self.multihead_attention = MultiHeadSelfAttention(n_heads, d_embedding, bias)
        self.layer_norm1 = nn.LayerNorm(d_embedding)
        self.feed_forward = FeedForward(d_embedding, 4 * d_embedding)
        self.layer_norm2 = nn.LayerNorm(d_embedding)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, x, mask = None):
        temp = x + self.dropout(self.multihead_attention(self.layer_norm1(x), mask))
        temp = temp + self.dropout(self.feed_forward(self.layer_norm2(temp)))
        return temp


class DecoderLayer(nn.Module):
    def __init__(self, n_heads, d_embedding, bias, p_drop):
        super().__init__()
        self.multihead_attention = MultiHeadSelfAttention(n_heads, d_embedding, bias)
        self.layer_norm1 = nn.LayerNorm(d_embedding)
        self.feed_forward = FeedForward(d_embedding, 4 * d_embedding)
        self.layer_norm2 = nn.LayerNorm(d_embedding)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, x, mask = None):
        temp = self.layer_norm1(self.dropout(self.multihead_attention(x, mask)) + x)
        temp = self.layer_norm2(self.dropout(self.feed_forward(temp)) + temp)
        return temp


class DecoderLayerEncDec(nn.Module):
    def __init__(self, n_heads, d_embedding, bias, p_drop):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(n_heads, d_embedding, bias)
        self.ln1 = nn.LayerNorm(d_embedding)

        self.cross_attn = MultiHeadCrossAttention(n_heads, d_embedding, bias)
        self.ln2 = nn.LayerNorm(d_embedding)

        self.ff = FeedForward(d_embedding, 4 * d_embedding)
        self.ln3 = nn.LayerNorm(d_embedding)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, x, enc_out, self_mask=None, cross_mask=None):
        temp = x + self.dropout(self.self_attn(self.ln1(x), self_mask))
        temp = temp + self.dropout(self.cross_attn(enc_out, self.ln2(temp), cross_mask))
        temp = temp + self.dropout(self.ff(self.ln3(temp)))
        return temp


class LMHeadWrapper(nn.Module):
    def __init__(self, model, vocab_size, embedding, d_embedding, weight_tying=True):
        super().__init__()
        self.model = model
        self.lm_head = nn.Linear(d_embedding, vocab_size, bias=False)

        if weight_tying:
            self.lm_head.weight = embedding.weight

    def forward(self, *args, **kwargs):
        hidden = self.model(*args, **kwargs)
        return self.lm_head(hidden)
