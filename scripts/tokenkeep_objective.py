"""Fixed retention change; same RNG draws and unscaled masked tokens."""
import torch
KEEP_MIN=0.5
def mask(batch,tokens,keep_min=KEEP_MIN,generator=None):
    assert 0<=keep_min<=1
    ratios=torch.rand(batch,1,generator=generator)
    if keep_min:ratios=keep_min+(1-keep_min)*ratios
    return torch.rand(batch,tokens,generator=generator)<=ratios
def apply(tokens,keep_min=KEEP_MIN):
    b,n,_=tokens.shape
    return tokens*mask(b,n,keep_min).to(tokens.device).unsqueeze(-1)
