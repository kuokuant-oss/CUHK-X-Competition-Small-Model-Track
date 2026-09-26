"""One lossless F0 base, one int8 ViT encoder, FP32 learned heads."""
# Role: layout of the pack inside the outer envelope (member base, int8 ViT encoder stored as a
# zlib-compressed torch file, readout heads, metadata) and the general manifest functions for it.
# Used by: fd18_codec (state_spec and check_state when loading; encode when writing the checkpoint);
# inference. load, predict_head, member and fuse are not called by the delivered run.
import io,zlib,json,hashlib
import numpy as np
from pathlib import Path
import torch
from cuhkx.fd13_codec import restore as restore_base
from cuhkx.fd_codec import tensor_spec
from cuhkx.fd15_vit import dequantize

def state_spec(x):
    # Manifest of a nested value: a tensor becomes its dtype, shape, element count and SHA256
    # (fd_codec.tensor_spec); dicts, lists and tuples recurse; any other value is kept as is, so
    # comparing two manifests also compares strings, numbers and flags exactly.
    if isinstance(x,torch.Tensor):return {'tensor':tensor_spec(x)}
    if isinstance(x,dict):return {'dict':{k:state_spec(v) for k,v in x.items()}}
    if isinstance(x,(list,tuple)):return {'sequence':[state_spec(v) for v in x]}
    return {'value':x}

def check_state(x,spec):
    # Recompute the manifest and compare it as a whole; any difference raises AssertionError.
    assert state_spec(x)==spec,'FD16 tensor/metadata identity mismatch'

def encode(base,encoder,heads,meta):
    # The encoder state is serialised with torch.save and compressed with zlib level 6; its
    # manifest is taken from the uncompressed state, so the reader checks the decoded tensors.
    stream=io.BytesIO();torch.save(encoder,stream)
    compressed=torch.from_numpy(np.frombuffer(zlib.compress(stream.getvalue(),6),dtype=np.uint8).copy())
    # The member base is stored as given (already encoded); meta_sha256 is the SHA256 of the
    # metadata as JSON with sorted keys.
    return dict(format='cuhkx-fd16-v1',base=base,encoder=compressed,encoder_manifest=state_spec(encoder),heads=heads,heads_manifest=state_spec(heads),meta=meta,meta_sha256=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest())

def load(path):
    # Reader for a file that holds this layout directly (no outer envelope and no manifest of the
    # encoded member base); the delivered checkpoint is read by fd18_codec.load instead.
    p=torch.load(path,map_location='cpu',weights_only=True)
    assert p['format']=='cuhkx-fd16-v1' and p['meta_sha256']==hashlib.sha256(json.dumps(p['meta'],sort_keys=True).encode()).hexdigest()
    # Inflate the encoder, check encoder and heads against their manifests, decode the member base.
    e=torch.load(io.BytesIO(zlib.decompress(p['encoder'].numpy().tobytes())),map_location='cpu',weights_only=True);check_state(e,p['encoder_manifest']);check_state(p['heads'],p['heads_manifest']);members,det=restore_base(p['base'])
    return p,e,p['heads'],members,det

def predict_head(features,h):
    # FP32 deployed scaler/head and explicit 0..39 class mapping.
    # Identical to fd15_codec.predict_head, the copy the delivered readout calls: standardise with
    # the head's mean and scale, apply the linear layer and softmax, then write the probabilities
    # into the columns h['classes'] of a 40-class output (other columns stay 0).
    x=torch.as_tensor(features,dtype=torch.float32)
    q=torch.softmax(torch.nn.functional.linear((x-h['mean'])/h['scale'],h['weight'],h['bias']),-1)
    out=torch.zeros((*q.shape[:-1],40),dtype=torch.float32);out[...,h['classes'].long()]=q
    return out

def member(features,heads,mode):
    # Readout probabilities for modality I, D or both ('ID'): each head's output is averaged over
    # axis 1 (the flip axis of the features built by el22_support_inference), then over modalities.
    names=['I','D'] if mode=='ID' else [mode]
    return torch.stack([predict_head(features[n],heads[n]).mean(1) for n in names]).mean(0)

def fuse(f0,vi,vd,mode,available_i,available_d):
    # Arithmetic blend 0.8 F0 + 0.2 v, where v is the mean of the readouts of the modalities that
    # `mode` selects and that are available for the clip; clips with none of them keep F0.
    import numpy as np
    f0=np.asarray(f0,dtype=np.float64);n=len(f0);v=np.zeros_like(f0);count=np.zeros(n)
    for key,q,mask in [('I',vi,available_i),('D',vd,available_d)]:
        if mode not in (key,'ID'):continue
        mask=np.asarray(mask,dtype=bool)
        if q is not None:v[mask]+=np.asarray(q,dtype=np.float64)[mask];count[mask]+=1
    alive=count>0;v[alive]/=count[alive,None];out=f0.copy();out[alive]=.8*f0[alive]+.2*v[alive]
    return out
