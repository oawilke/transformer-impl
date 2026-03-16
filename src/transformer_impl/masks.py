import torch

def src_keep(src, pad_id):
    return (src != pad_id)[:, None, :]

def tgt_self_keep(tgt_inp, pad_id):
    B, T = tgt_inp.shape
    pad_keep = (tgt_inp != pad_id)[:, None, :]
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=tgt_inp.device))[None,:,:]
    return causal & pad_keep.expand(B, T, T)

def cross_keep(tgt_inp, src, pad_id):
    B, Tt = tgt_inp.shape
    return (src != pad_id)[:, None, :].expand(B, Tt, src.shape[1])