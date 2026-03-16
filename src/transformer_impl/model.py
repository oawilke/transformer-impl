import torch.nn as nn

from .embeddings import InputEmbedding, PositionalEmbedding
from .layers import DecoderLayerEncDec, EncoderLayer, DecoderLayer


class EncoderOnlyTransformer(nn.Module):
    """Encoder-only Transformer."""
    def __init__(self, n_layers, n_heads, d_embedding, vocab_size, max_len, bias, p_drop=0.1):
        super().__init__()

        self.input_embedding = InputEmbedding(vocab_size, d_embedding)
        self.positional_embedding = PositionalEmbedding(max_len, d_embedding)

        self.layers = nn.ModuleList(
            [EncoderLayer(n_heads, d_embedding, bias, p_drop) for _ in range(n_layers)]
        )

        self.dropout = nn.Dropout(p_drop)

    def forward(self, x, mask = None):
        inp = self.input_embedding(x) + self.positional_embedding(x)
        val = self.dropout(inp)

        for layer in self.layers:
            val = layer(val, mask)

        return val


class DecoderOnlyTransformer(nn.Module):
    """Decoder-only Transformer."""
    def __init__(self, n_layers, n_heads, d_embedding, vocab_size, max_len, bias, p_drop=0.1):
        super().__init__()
        self.input_embedding = InputEmbedding(vocab_size, d_embedding)
        self.positional_embedding = PositionalEmbedding(max_len, d_embedding)

        self.layers = nn.ModuleList(
            [DecoderLayer(n_heads, d_embedding, bias, p_drop) for _ in range(n_layers)]
        )

        self.dropout = nn.Dropout(p_drop)

    def forward(self, x, mask = None):
        h = self.input_embedding(x) + self.positional_embedding(x)
        h = self.dropout(h)

        for layer in self.layers:
            h = layer(h, mask=mask)

        return h


class CrossAttentionDecoderTransformer(nn.Module):
    """Decoder of an encoder-decoder Transformer."""
    def __init__(self, n_layers, n_heads, d_embedding, vocab_size, max_len, bias, p_drop=0.1):
        super().__init__()
        self.input_embedding = InputEmbedding(vocab_size, d_embedding)
        self.positional_embedding = PositionalEmbedding(max_len, d_embedding)

        self.layers = nn.ModuleList(
            [DecoderLayerEncDec(n_heads, d_embedding, bias, p_drop) for _ in range(n_layers)]
        )

        self.dropout = nn.Dropout(p_drop)

    def forward(self, x, enc_out, self_mask=None, cross_mask=None):
        h = self.input_embedding(x) + self.positional_embedding(x)
        h = self.dropout(h)

        for layer in self.layers:
            h = layer(h, enc_out, self_mask=self_mask, cross_mask=cross_mask)

        return h


class EncoderDecoderTransformer(nn.Module):
    """Encoder-decoder Transformer."""
    def __init__(self, n_layers_encoder, n_layers_decoder, n_heads, d_embedding, bias, max_len_src, max_len_target, vocab_size, p_drop=0.1):
        super().__init__()

        self.input_embedding = InputEmbedding(vocab_size, d_embedding)

        self.encoder = EncoderOnlyTransformer(
            n_layers_encoder, n_heads, d_embedding, vocab_size, max_len_src, bias, p_drop
        )
        self.encoder.input_embedding = self.input_embedding

        self.decoder = CrossAttentionDecoderTransformer(
            n_layers_decoder, n_heads, d_embedding, vocab_size, max_len_target, bias, p_drop
        )
        self.decoder.input_embedding = self.input_embedding

        self.lm_head = nn.Linear(d_embedding, vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.input_embedding.embedding.weight

    def forward(self, src, tgt_inp, src_mask=None, tgt_self_mask=None, tgt_cross_mask=None):
        enc_out = self.encoder(src, mask=src_mask)
        dec_out = self.decoder(tgt_inp, enc_out, self_mask=tgt_self_mask, cross_mask=tgt_cross_mask)
        return self.lm_head(dec_out)
