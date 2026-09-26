"""Exact int8 cross-checkpoint deltas in an ordinary deflated torch ZIP."""
# Role: per-tensor SHA256 manifests (tensor_spec, state_spec, check_state) and fp32(), which turns
# a stored member state into FP32 tensors. encode_delta to load form a whole-pack format of their
# own (mod-256 int8 deltas between members) that the delivered checkpoint does not use.
# Used by: fd16_codec, fd13_codec, fd_lzma_codec, fd_entropy_codec, fd_mixed_codec (manifests),
# fd13_inference.model_from_spec (fp32) and the inference entry point (check_state); inference.
from __future__ import annotations

import hashlib
import io
import math
import zipfile
import zlib
from pathlib import Path

import numpy as np
import torch

# Format tag of this module's whole-pack format (assemble/restore); not the delivered checkpoint's.
FORMAT = 'cuhkx-fd-lossless-v1'


def tensor_spec(t):
    # Manifest entry of one CPU tensor: dtype, shape, element count and the SHA256 of its bytes in
    # row-major order.
    assert isinstance(t,torch.Tensor) and t.device.type=='cpu'
    a=t.detach().contiguous().numpy()
    return dict(dtype=str(t.dtype),shape=list(t.shape),elements=t.numel(),sha256=hashlib.sha256(a.tobytes()).hexdigest())


def state_spec(x):
    # Nested dicts of tensors only; fd16_codec.state_spec also accepts lists and plain values.
    if isinstance(x,dict): return {k:state_spec(v) for k,v in x.items()}
    return tensor_spec(x)


def check_state(x, spec):
    # Compare with a manifest level by level: same keys everywhere and equal tensor entries;
    # raises ValueError at the first difference.
    if isinstance(x,dict):
        if set(x)!=set(spec): raise ValueError('Encoded state keys differ')
        for k,v in x.items(): check_state(v,spec[k])
    elif tensor_spec(x)!=spec: raise ValueError('Tensor dtype/shape/elements/SHA mismatch')


# Whole-pack format of this module (encode_delta to load): members stored as they are or as deltas
# against an earlier member, plus the detector and manifests. The delivered checkpoint does not use
# it; its member C is stored by fd_lzma_codec.encode_delta.
def encode_delta(base,target):
    # For each int8 pair: (target codes - base codes) mod 256 as bytes, zlib level 6, with the
    # target's scale and manifests of the base codes, target codes and scale; other entries are
    # kept as they are.
    if set(base)!=set(target): raise ValueError('Delta requires identical state keys')
    out={}
    for k,v in target.items():
        b=base[k]
        if isinstance(v,dict) and set(v)=={'_int8','_scale'} and isinstance(b,dict) and set(b)=={'_int8','_scale'}:
            a,q=b['_int8'],v['_int8']
            if a.dtype!=torch.int8 or q.dtype!=torch.int8 or a.shape!=q.shape: raise ValueError('Incompatible integer tensors')
            residual=((q.numpy().astype(np.int16)-a.numpy().astype(np.int16))%256).astype(np.uint8)
            blob=zlib.compress(residual.tobytes(),6)
            out[k]=dict(_delta_mod256_zlib=torch.from_numpy(np.frombuffer(blob,dtype=np.uint8).copy()),
                _scale=v['_scale'].clone(),shape=list(q.shape),elements=q.numel(),base_spec=tensor_spec(a),target_spec=tensor_spec(q),scale_spec=tensor_spec(v['_scale']))
        else:
            out[k]=v
    return out


def restore_delta(base,delta):
    # Inverse of encode_delta, checking the schema, the reference codes and stored scale against
    # their manifests, the shapes, an exact-length inflate and the rebuilt codes' manifest.
    # (a + r) mod 256 read as int8 is exactly the target code, because codes lie in -128..127.
    if set(base)!=set(delta): raise ValueError('Delta keys do not match reference')
    out={}
    required={'_delta_mod256_zlib','_scale','shape','elements','base_spec','target_spec','scale_spec'}
    for k,v in delta.items():
        if isinstance(v,dict) and '_delta_mod256_zlib' in v:
            if set(v)!=required: raise ValueError('Invalid delta schema')
            b=base[k]
            if not isinstance(b,dict) or set(b)!= {'_int8','_scale'}: raise ValueError('Delta reference is not int8')
            a=b['_int8']; n=v['elements']
            if tensor_spec(a)!=v['base_spec'] or tensor_spec(v['_scale'])!=v['scale_spec']: raise ValueError('Reference/scale hash mismatch')
            if a.dtype!=torch.int8 or list(a.shape)!=v['shape'] or n!=a.numel() or n!=math.prod(v['shape']): raise ValueError('Delta shape mismatch')
            blob=v['_delta_mod256_zlib']
            if blob.dtype!=torch.uint8 or blob.ndim!=1: raise ValueError('Invalid compressed bytes')
            d=zlib.decompressobj(); raw=d.decompress(blob.numpy().tobytes(),n+1)
            if len(raw)!=n or not d.eof or d.unused_data or d.unconsumed_tail: raise ValueError('Truncated, oversized or trailing compressed data')
            residual=np.frombuffer(raw,dtype=np.uint8).reshape(v['shape'])
            q=((a.numpy().astype(np.int16)+residual.astype(np.int16))%256).astype(np.uint8).view(np.int8)
            recovered=torch.from_numpy(q.copy())
            if tensor_spec(recovered)!=v['target_spec']: raise ValueError('Restored integer SHA mismatch')
            out[k]={'_int8':recovered,'_scale':v['_scale']}
        else: out[k]=v
    return out


def assemble(members,detector,meta,references):
    # Members are encoded in order; references[name] is None (stored as is) or an earlier member
    # whose state is the base of the delta.
    order=list(members); encoded={}; restored={}
    for name in order:
        reference=references[name]
        if reference is not None and reference not in restored: raise ValueError('Missing/forward/cyclic reference')
        encoded[name]=members[name] if reference is None else encode_delta(restored[reference],members[name])
        restored[name]=members[name]
    return dict(format=FORMAT,version=1,meta=meta,member_order=order,references=references,
        components={'detector':detector,**encoded},tensor_manifest={'detector':state_spec(detector),**{k:state_spec(v) for k,v in members.items()}})


def restore(pack):
    # Check the schema and the detector, then rebuild each member in order from its reference and
    # check it against the manifest.
    if set(pack)!= {'format','version','meta','member_order','references','components','tensor_manifest'} or pack['format']!=FORMAT or pack['version']!=1: raise ValueError('Unknown FD checkpoint schema')
    names=pack['member_order']
    if len(names)!=len(set(names)) or set(pack['components'])!={'detector',*names} or set(pack['tensor_manifest'])!={'detector',*names} or set(pack['references'])!=set(names): raise ValueError('Invalid member schema')
    detector=pack['components']['detector']; check_state(detector,pack['tensor_manifest']['detector'])
    members={}
    for name in names:
        ref=pack['references'][name]
        if ref is not None and ref not in members: raise ValueError('Invalid reference order')
        members[name]=pack['components'][name] if ref is None else restore_delta(members[ref],pack['components'][name])
        check_state(members[name],pack['tensor_manifest'][name])
    return members,detector


def serialize(pack,deflate=True):
    # torch.save the pack; if deflate, re-write each entry of the ZIP archive with DEFLATE level 6.
    raw=io.BytesIO(); torch.save(pack,raw)
    if not deflate: return raw.getvalue()
    raw.seek(0); result=io.BytesIO()
    with zipfile.ZipFile(raw,'r') as src,zipfile.ZipFile(result,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as dst:
        for info in src.infolist():
            dst.writestr(info.filename,src.read(info.filename),compress_type=zipfile.ZIP_DEFLATED,compresslevel=6)
    return result.getvalue()


def load(path):
    # Ordinary stream loading supports compressed storage; mmap is deliberately absent.
    with Path(path).open('rb') as f: pack=torch.load(f,map_location='cpu',weights_only=True)
    members,detector=restore(pack)
    return pack,members,detector


# Used at inference for members B and C (fd13_inference.model_from_spec).
def fp32(state):
    # Stored member state -> FP32 state for load_state_dict: if any entry is quantised (a dict),
    # budget.dequantize_state rebuilds it as codes x scale and casts the other floating tensors to
    # FP32; the last line casts any floating tensor left to FP32 and keeps integer tensors.
    from cuhkx.budget import dequantize_state
    if any(isinstance(v,dict) for v in state.values()): state=dequantize_state(state)
    return {k:v.float() if v.is_floating_point() else v for k,v in state.items()}
