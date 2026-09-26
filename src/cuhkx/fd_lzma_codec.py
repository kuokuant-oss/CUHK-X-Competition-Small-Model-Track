"""Lossless per-tensor entropy storage; selection uses bytes, never labels."""
# Role: lossless per-tensor storage inside the member base: each tensor is kept as zlib or LZMA
# bytes, whichever is smaller, with its SHA256 manifest; member C is kept as mod-256 residuals of
# its int8 codes against member B's, which restore_delta turns back into C exactly.
# Used by: fd13_codec.restore (restore_state, restore_delta) and the inference entry point's
# detector-only step (restore_state); inference. The encode_* functions build these entries.
# Not used by the delivered pack: the 7-bit packed path (encode_thermal, restore_thermal;
# fd13_codec reaches it only for optional members, and the pack has none) and
# assemble/restore/load, a whole-pack format with its own tag.
import lzma
import zlib
from pathlib import Path
import numpy as np
import torch
# Bit packing helpers; only the 7-bit packed path uses them.
from cuhkx.budget import _bit_pack, _bit_unpack
from cuhkx.fd_codec import tensor_spec, state_spec, check_state
# prediction(): B's int8 codes rescaled to C's scale, the predictor of 'scale-context-residual'.
from cuhkx.fd_entropy_codec import prediction

# Format tag of this module's whole-pack format (assemble/restore); not the delivered one.
FORMAT='cuhkx-fd-lzma-byteplane-v2'


def encode_tensor(t):
    # Candidate byte layouts: the raw bytes ('identity') and, for dtypes wider than one byte,
    # 'byteplanes': byte 0 of every element, then byte 1, and so on, so that bytes of equal
    # significance sit together (for floats, the high bytes that hold sign and exponent).
    raw=t.detach().contiguous().numpy().tobytes()
    transforms=[('identity',raw)]
    if t.element_size()>1 and t.numel():
        transforms.append(('byteplanes',np.frombuffer(raw,dtype=np.uint8).reshape(-1,t.element_size()).T.copy().tobytes()))
    # Compress each layout with zlib (level 6) and LZMA (preset 6) and keep the smallest blob;
    # equal sizes are broken by transform name, then codec name, so the choice is deterministic.
    options=[(transform,codec,blob) for transform,data in transforms for codec,blob in [('zlib6',zlib.compress(data,6)),('lzma6',lzma.compress(data,preset=6))]]
    transform,codec,blob=min(options,key=lambda v:(len(v[2]),v[0],v[1]))
    # The spec (dtype, shape, element count, SHA256 of the original bytes) is checked on restore.
    return dict(kind='tensor',spec=tensor_spec(t),transform=transform,codec=codec,blob=torch.from_numpy(np.frombuffer(blob,dtype=np.uint8).copy()))


def restore_tensor(e):
    # Check the schema, the dtype (from a fixed list) and that shape and element count agree.
    if set(e)!={'kind','spec','transform','codec','blob'} or e['kind']!='tensor':raise ValueError('Invalid compressed tensor schema')
    spec=e['spec'];dtypes={'torch.int8':np.int8,'torch.uint8':np.uint8,'torch.float16':np.float16,'torch.float32':np.float32,'torch.float64':np.float64,'torch.int16':np.int16,'torch.int32':np.int32,'torch.int64':np.int64,'torch.bool':np.bool_}
    if set(spec)!={'dtype','shape','elements','sha256'} or spec['dtype'] not in dtypes:raise ValueError('Invalid tensor metadata')
    shape=spec['shape'];count=spec['elements'];dtype=np.dtype(dtypes[spec['dtype']])
    if not isinstance(count,int) or count<0 or any(not isinstance(v,int) or v<0 for v in shape) or int(np.prod(shape))!=count:raise ValueError('Invalid tensor size')
    # n is the expected number of bytes; the blob must be a flat uint8 tensor.
    n=count*dtype.itemsize;blob=e['blob']
    if blob.dtype!=torch.uint8 or blob.ndim!=1:raise ValueError('Invalid compressed bytes')
    # Inflate at most n + 1 bytes and require a complete stream with nothing after it (for zlib
    # also no input held back; LZMA runs with a 128 MiB memory limit), then exactly n bytes out.
    if e['codec']=='zlib6':
        d=zlib.decompressobj();raw=d.decompress(blob.numpy().tobytes(),n+1)
        if not d.eof or d.unused_data or d.unconsumed_tail:raise ValueError('Invalid zlib stream')
    elif e['codec']=='lzma6':
        d=lzma.LZMADecompressor(memlimit=128*1024*1024);raw=d.decompress(blob.numpy().tobytes(),max_length=n+1)
        if not d.eof or d.unused_data:raise ValueError('Invalid LZMA stream')
    else:raise ValueError('Unknown entropy codec')
    if len(raw)!=n:raise ValueError('Truncated or oversized tensor')
    # Undo the byte-plane layout: (itemsize, count) bytes back to (count, itemsize).
    if e['transform']=='byteplanes':
        if dtype.itemsize<=1 or count==0:raise ValueError('Invalid byteplane transform')
        raw=np.frombuffer(raw,dtype=np.uint8).reshape(dtype.itemsize,count).T.copy().tobytes()
    elif e['transform']!='identity':raise ValueError('Unknown byte transform')
    # Rebuild the tensor and check dtype, shape, element count and SHA256 against the spec.
    value=torch.from_numpy(np.frombuffer(raw,dtype=dtype).reshape(shape).copy())
    if tensor_spec(value)!=spec:raise ValueError('Restored tensor SHA mismatch')
    return value


def encode_state(s):
    # Nested dict (e.g. a member state with {'_int8', '_scale'} pairs) -> the same nesting with
    # every leaf tensor compressed on its own by encode_tensor.
    if isinstance(s,dict):return dict(kind='state',values={k:encode_state(v) for k,v in s.items()})
    return encode_tensor(s)


def restore_state(e):
    # Inverse of encode_state; restore_tensor checks every leaf against its SHA256.
    if e['kind']=='state':
        if set(e)!={'kind','values'}:raise ValueError('Invalid encoded state')
        return {k:restore_state(v) for k,v in e['values'].items()}
    return restore_tensor(e)


def encode_delta(base,target):
    # Encode the target state (member C) against the base state (member B). For each int8 pair
    # ({'_int8', '_scale'}) two residuals of C's codes are tried and the one with the smaller
    # compressed payload is kept:
    #   'original-residual': (C codes - B codes) mod 256, in the original element order;
    #   'scale-context-residual': (C codes - prediction) mod 256, where the prediction is B's
    #   codes rescaled to C's scale (fd_entropy_codec.prediction), reordered by a stable sort of
    #   B's codes.
    # Both depend only on B and on C's stored scale, so restore_delta can redo them. Entries that
    # are not int8 pairs are kept verbatim, uncompressed here (the outer zlib envelope covers
    # them). `stats` lists the chosen mode and sizes per tensor.
    if set(base)!=set(target):raise ValueError('Different state keys')
    out={};stats=[]
    for key,value in target.items():
        if isinstance(value,dict) and set(value)=={'_int8','_scale'}:
            b=base[key];q=value['_int8']
            if set(b)!={'_int8','_scale'} or q.dtype!=torch.int8 or b['_int8'].dtype!=torch.int8 or q.shape!=b['_int8'].shape:raise ValueError('Invalid reference state')
            # int16 avoids overflow in the subtraction; % 256 maps the difference to 0..255 (uint8).
            direct=((q.numpy().astype(np.int16)-b['_int8'].numpy().astype(np.int16))%256).astype(np.uint8)
            scaled=((q.numpy().astype(np.int16)-prediction(b,value['_scale']))%256).astype(np.uint8).reshape(-1)
            # Stable sort of B's codes: weights with equal B code become neighbours in the stream.
            order=np.argsort(b['_int8'].numpy().reshape(-1),kind='stable')
            # encode_tensor compresses each flat residual with zlib or LZMA; equal sizes go to
            # 'original-residual' (the alphabetically first mode).
            options=[('original-residual',encode_tensor(torch.from_numpy(direct.reshape(-1).copy()))),('scale-context-residual',encode_tensor(torch.from_numpy(scaled[order].copy())))]
            mode,payload=min(options,key=lambda v:(v[1]['blob'].numel(),v[0]))
            # Keep C's scale as is, plus manifests of B's pair (to check the reference on restore)
            # and of C's pair (to check the result).
            out[key]=dict(kind='residual',mode=mode,payload=payload,scale=value['_scale'],base_spec=state_spec(b),target_spec=state_spec(value))
            stats.append(dict(key=key,mode=mode,codec=payload['codec'],bytes=payload['blob'].numel(),alternatives={k:dict(codec=v['codec'],bytes=v['blob'].numel()) for k,v in options}))
        else:out[key]=dict(kind='verbatim',value=value,target_spec=state_spec(value))
    return out,stats


def restore_delta(base,encoded):
    # Rebuild member C from member B and the entries of encode_delta; each rebuilt entry is checked
    # against C's manifest (target_spec) before it is returned.
    if set(base)!=set(encoded):raise ValueError('Reference keys differ')
    out={}
    for key,e in encoded.items():
        if e['kind']=='verbatim':
            if set(e)!={'kind','value','target_spec'}:raise ValueError('Invalid verbatim schema')
            value=e['value']
        elif e['kind']=='residual':
            if set(e)!={'kind','mode','payload','scale','base_spec','target_spec'}:raise ValueError('Invalid residual schema')
            # Check B's pair against the reference manifest, then decompress the residual (one uint8
            # per element). Note that q holds B's codes here, not C's.
            b=base[key];check_state(b,e['base_spec']);q=b['_int8'];res=restore_tensor(e['payload'])
            if res.dtype!=torch.uint8 or res.ndim!=1 or res.numel()!=q.numel():raise ValueError('Invalid residual length')
            residual=res.numpy()
            if e['mode']=='scale-context-residual':
                # Undo the reordering (stored item i belongs to element order[i]), then recompute
                # the prediction from B's pair and C's stored scale.
                order=np.argsort(q.numpy().reshape(-1),kind='stable');unsorted=np.empty(q.numel(),dtype=np.uint8);unsorted[order]=residual;residual=unsorted
                pred=prediction(b,e['scale'])
            # For 'original-residual' the prediction is B's codes themselves.
            elif e['mode']=='original-residual':pred=q.numpy().astype(np.int16)
            else:raise ValueError('Unknown compression prediction')
            # Mod-256 arithmetic: residual = (C - pred) mod 256, so (pred + residual) mod 256 equals
            # C mod 256, and that byte read as int8 (two's complement) is C's code exactly, since
            # int8 codes lie in -128..127. The prediction affects only how well residuals compress.
            restored=((pred+residual.reshape(q.shape).astype(np.int16))%256).astype(np.uint8).view(np.int8)
            value=dict(_int8=torch.from_numpy(restored.copy()),_scale=e['scale'])
        else:raise ValueError('Unknown residual kind')
        # Check against C's manifest entry (dtype, shape, element count and SHA256 of each tensor).
        check_state(value,e['target_spec']);out[key]=value
    return out


# 7-bit packed states (as written by budget.quantize_state with bits=7). Despite 'thermal' in the
# names, these functions accept any state with 7-bit packed entries. The delivered pack stores
# none: B and C are int8, and fd13_codec calls restore_thermal only for optional members, of
# which the pack has none.
def encode_thermal(state):
    out={}
    for key,value in state.items():
        if isinstance(value,dict) and '_packed' in value:
            if set(value)!={'_packed','_bits','_shape','_scale'} or int(value['_bits'])!=7:raise ValueError('Invalid thermal packed state')
            # n codes of 7 bits; `used` = data bits in the last packed byte (0 if the byte is full).
            n=int(np.prod(value['_shape'].numpy()));packed=value['_packed'];used=n*7%8
            if packed.dtype!=torch.uint8 or packed.ndim!=1 or packed.numel()!=(n*7+7)//8:raise ValueError('Invalid packed size')
            # Unpack to one byte per code and keep the unused high bits of the last packed byte as
            # `padding`, so that restore_thermal rebuilds the packed bytes bit for bit.
            codes=_bit_unpack(packed.numpy(),n,7);padding=int(packed[-1])&(255^((1<<used)-1)) if used else 0
            out[key]=dict(kind='unsigned7',payload=encode_tensor(torch.from_numpy(codes.copy())),padding=padding,meta={k:v for k,v in value.items() if k!='_packed'},target_spec=state_spec(value))
        else:out[key]=dict(kind='verbatim',value=value,target_spec=state_spec(value))
    return out


def restore_thermal(encoded):
    # Inverse of encode_thermal: decode the codes (each at most 127), pack them again at 7 bits,
    # put the padding bits back into the last byte and check the original manifest.
    out={}
    for key,e in encoded.items():
        if e['kind']=='verbatim':
            if set(e)!={'kind','value','target_spec'}:raise ValueError('Invalid thermal verbatim schema')
            value=e['value']
        elif e['kind']=='unsigned7':
            if set(e)!={'kind','payload','padding','meta','target_spec'}:raise ValueError('Invalid unsigned7 schema')
            meta=e['meta']
            if set(meta)!={'_bits','_shape','_scale'} or int(meta['_bits'])!=7:raise ValueError('Invalid thermal metadata')
            shape=tuple(map(int,meta['_shape']));n=int(np.prod(shape));codes=restore_tensor(e['payload'])
            if not shape or min(shape)<=0 or codes.dtype!=torch.uint8 or codes.ndim!=1 or codes.numel()!=n or (codes.numpy()>127).any():raise ValueError('Invalid unsigned7 codes')
            used=n*7%8;mask=255^((1<<used)-1) if used else 0;pad=e['padding']
            if not isinstance(pad,int) or pad<0 or pad>255 or pad&~mask:raise ValueError('Invalid original padding')
            packed=_bit_pack(codes.numpy(),7)
            if used:packed[-1]|=pad
            value=dict(_packed=torch.from_numpy(packed.copy()),**meta)
        else:raise ValueError('Unknown thermal kind')
        check_state(value,e['target_spec']);out[key]=value
    return out


# Whole-pack format of this module (B, C as a delta, a 7-bit packed state and the detector). The
# delivered checkpoint uses fd13_codec's member base instead; nothing in this package calls
# assemble, restore or load below.
def assemble(base,target,thermal,detector,meta):
    delta,stats=encode_delta(base,target)
    return dict(format=FORMAT,version=1,meta=meta,base=encode_state(base),C_delta=delta,thermal_codes=encode_thermal(thermal),detector=encode_state(detector),
        tensor_manifest={k:state_spec(v) for k,v in dict(B=base,C=target,thermal=thermal,detector=detector).items()}),stats


def restore(pack):
    if set(pack)!={'format','version','meta','base','C_delta','thermal_codes','detector','tensor_manifest'} or pack['format']!=FORMAT or pack['version']!=1:raise ValueError('Invalid LZMA checkpoint')
    m=pack['tensor_manifest']
    if set(m)!={'B','C','thermal','detector'}:raise ValueError('Invalid manifest')
    # Decode and check B and the detector, then rebuild C and the packed state and check them too.
    base=restore_state(pack['base']);det=restore_state(pack['detector']);check_state(base,m['B']);check_state(det,m['detector'])
    members=dict(B=base,C=restore_delta(base,pack['C_delta']),thermal=restore_thermal(pack['thermal_codes']))
    for k,v in members.items():check_state(v,m[k])
    return members,det


def load(path):
    with Path(path).open('rb') as f:pack=torch.load(f,map_location='cpu',weights_only=True)
    members,detector=restore(pack);return pack,members,detector
