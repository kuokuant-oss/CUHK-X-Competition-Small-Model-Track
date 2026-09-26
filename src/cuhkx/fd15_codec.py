"""One lossless F0 base, one int8 ViT encoder, FP32 learned heads."""
# Role: a checkpoint layout (format 'cuhkx-fd15-v1') that the delivered weights/model.pt does not
#   use, plus predict_head(), the FP32 readout head that the delivered run does use.
# Used by: fd18_readout and fd16_readout (predict_head), training/fd17_heads.py and
#   training/fd18_heads.py (predict_head); both. encode, load, member and fuse are not used by
#   the delivered run.
import io,zlib,json,hashlib
import numpy as np
from pathlib import Path
import torch
from cuhkx.fd13_codec import restore as restore_base
from cuhkx.fd_codec import check_state,state_spec
from cuhkx.fd15_vit import dequantize

# Packs the member base, the int8 encoder (torch.save, then zlib level 6), the heads and the
# metadata, with SHA256 manifests of the encoder and the heads and a SHA256 of the metadata.
def encode(base,encoder,heads,meta):
    stream=io.BytesIO();torch.save(encoder,stream)
    compressed=torch.from_numpy(np.frombuffer(zlib.compress(stream.getvalue(),6),dtype=np.uint8).copy())
    return dict(format='cuhkx-fd15-v1',base=base,encoder=compressed,encoder_manifest=state_spec(encoder),heads=heads,heads_manifest=state_spec(heads),meta=meta,meta_sha256=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest())

# Reads a checkpoint in this layout, checks the metadata hash and the manifests, and restores
# the members and the detector from the base.
def load(path):
    p=torch.load(path,map_location='cpu',weights_only=True)
    assert p['format']=='cuhkx-fd15-v1' and p['meta_sha256']==hashlib.sha256(json.dumps(p['meta'],sort_keys=True).encode()).hexdigest()
    e=torch.load(io.BytesIO(zlib.decompress(p['encoder'].numpy().tobytes())),map_location='cpu',weights_only=True);check_state(e,p['encoder_manifest']);check_state(p['heads'],p['heads_manifest']);members,det=restore_base(p['base'])
    return p,e,p['heads'],members,det

# `features`: (..., n) head inputs; `h`: a stored head with 'mean' and 'scale' (length n),
# 'weight', 'bias' and 'classes'. Standardise, apply the linear layer and softmax, then place
# each output column at its class index in a 40-column result (a class the head lacks keeps
# probability 0).
def predict_head(features,h):
    # FP32 deployed scaler/head and explicit 0..39 class mapping.
    x=torch.as_tensor(features,dtype=torch.float32)
    q=torch.softmax(torch.nn.functional.linear((x-h['mean'])/h['scale'],h['weight'],h['bias']),-1)
    out=torch.zeros((*q.shape[:-1],40),dtype=torch.float32);out[...,h['classes'].long()]=q
    return out

# Mean of the view-averaged probabilities of the per-modality heads 'I', 'D' or both ('ID').
def member(features,heads,mode):
    names=['I','D'] if mode=='ID' else [mode]
    return torch.stack([predict_head(features[n],heads[n]).mean(1) for n in names]).mean(0)

# Arithmetic fusion 0.8 F0 + 0.2 v, where v is the mean of the available per-modality readouts
# selected by `mode`; clips with none available keep F0.
def fuse(f0,vi,vd,mode,available_i,available_d):
    import numpy as np
    f0=np.asarray(f0,dtype=np.float64);n=len(f0);v=np.zeros_like(f0);count=np.zeros(n)
    for key,q,mask in [('I',vi,available_i),('D',vd,available_d)]:
        if mode not in (key,'ID'):continue
        mask=np.asarray(mask,dtype=bool)
        if q is not None:v[mask]+=np.asarray(q,dtype=np.float64)[mask];count[mask]+=1
    alive=count>0;v[alive]/=count[alive,None];out=f0.copy();out[alive]=.8*f0[alive]+.2*v[alive]
    return out
