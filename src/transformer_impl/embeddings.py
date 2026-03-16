import torch
import torch.nn as nn


# Segment embeddings are not considered in this implementation!
class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_embedding):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_embedding)

    def forward(self, x):
        return self.embedding(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_embedding):
        super().__init__()
        self.max_len = max_len
        self.emb = nn.Embedding(max_len, d_embedding)

    def forward(self, x):
        T = x.shape[1]
        if T > self.max_len:
            raise ValueError(f"Sequence length {T} exceeds max_len {self.max_len}")

        pos = torch.arange(T, device=x.device, dtype=torch.long)  # (T,)
        return self.emb(pos).unsqueeze(0)  # (1, T, d_embedding)
