"""VideoMAEv2 ViT-S adapter. Official architecture, optional verified SDPA."""
# Role: builds the frozen VideoMAEv2 ViT-S/16 from the upstream model code with fused attention,
#   prepares det248 frames for it, splits its forward pass after block 9, and converts its
#   weights to and from int8.
# Used by: el22_support_inference (make, dequantize, preprocess) and el22_support_pool (prefix);
#   inference.
from functools import partial
import types
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from cuhkx.fd15_official_vit import VisionTransformer

# Replacement forward for the upstream Attention module, using torch's fused
# scaled_dot_product_attention with the same projections, biases and scale and no attention
# dropout; it is mathematically equal to the upstream softmax(q k^T * scale) v.
def attention(self,x):
    b,n,c=x.shape
    # q and v have learned biases; the key bias is zero.
    bias=torch.cat((self.q_bias,torch.zeros_like(self.v_bias),self.v_bias)) if self.q_bias is not None else None
    # (batch, tokens, 3, heads, head dim) -> q, k, v, each (batch, heads, tokens, head dim).
    q,k,v=F.linear(x,self.qkv.weight,bias).reshape(b,n,3,self.num_heads,-1).permute(2,0,3,1,4).unbind(0)
    x=F.scaled_dot_product_attention(q,k,v,dropout_p=0.,scale=self.scale).transpose(1,2).reshape(b,n,-1)
    return self.proj_drop(self.proj(x))

# ViT-S/16 for 16-frame clips: width 384, 12 blocks, 6 heads, MLP ratio 4, tubelet 2, so a
# (3, 16, 224, 224) clip gives 8 x 14 x 14 = 1,568 tokens; mean pooling with a final LayerNorm
# (fc_norm). The `classes`-way head is not used by the readout. sdpa=True installs attention()
# as the forward of every block's attention module.
def make(classes=40,sdpa=True,head_drop=0.):
    m=VisionTransformer(patch_size=16,embed_dim=384,depth=12,num_heads=6,mlp_ratio=4,qkv_bias=True,norm_layer=partial(nn.LayerNorm,eps=1e-6),num_classes=classes,all_frames=16,tubelet_size=2,use_mean_pooling=True,head_drop_rate=head_drop)
    if sdpa:
        for b in m.blocks:b.attn.forward=types.MethodType(attention,b.attn)
    return m

def preprocess(x,modality,flip=False):
    # Raw det248 uint8 B,T,C,H,W; no resize, fixed center224.
    assert x.ndim==5 and x.shape[2]==4 and x.shape[-2:]==(248,248)
    # IR: channel 3 repeated to three channels; depth: the Depth_Color RGB channels 0-2.
    x=x[:,:,3:4].expand(-1,-1,3,-1,-1) if modality=='I' else x[:,:,:3]
    # Centre 224 crop (pixels 12-235), reordered to (B, 3, T, 224, 224) and scaled to [0, 1].
    x=x[:,:,:,12:236,12:236].permute(0,2,1,3,4).float()/255.
    # Horizontal flip along the width axis.
    if flip:x=x.flip(-1)
    # ImageNet mean and standard deviation.
    mean=x.new_tensor([.485,.456,.406]).view(1,3,1,1,1);std=x.new_tensor([.229,.224,.225]).view(1,3,1,1,1)
    return (x-mean)/std

# First part of the upstream forward pass: patch embedding, the fixed sine-cosine position
# table, blocks 0-9 (the upstream dropout after the position embedding has rate 0 and is left
# out). Returns the tokens, (batch, 1568, 384).
def prefix(m,x):
    x=m.patch_embed(x);x=x+m.pos_embed.expand(x.shape[0],-1,-1).type_as(x).to(x.device).clone().detach()
    for b in m.blocks[:10]:x=b(x)
    return x

# Rest of the upstream forward pass: blocks 10-11, mean pooling, fc_norm and the classification
# head. Not called in this package; the readout pools the last-block tokens itself.
def tail(m,x):
    for b in m.blocks[10:]:x=b(x)
    return m.head(m.head_dropout(m.fc_norm(x.mean(1))))

# Int8 storage: every tensor with two or more dimensions, except the classification head, gets
# symmetric per-output-channel codes in [-127, 127] with scale = max|w| / 127 over all but the
# first axis; all other tensors stay FP32. The encoder in weights/model.pt uses this format;
# this function is not called at inference.
def quantize(state):
    out={}
    for k,v in state.items():
        v=v.detach().cpu().float().contiguous()
        if v.ndim>=2 and not k.startswith('head.'):
            scale=(v.abs().amax(tuple(range(1,v.ndim)),keepdim=True)/127).clamp_min(1e-12)
            out[k]={'codes':(v/scale).round().clamp(-127,127).to(torch.int8),'scale':scale}
        else:out[k]=v
    return out

# Inverse of quantize(): codes x scale as FP32; the other tensors are copied.
def dequantize(q):
    return {k:v['codes'].float()*v['scale'] if isinstance(v,dict) else v.clone() for k,v in q.items()}
