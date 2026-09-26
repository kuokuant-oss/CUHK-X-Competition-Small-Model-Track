"""Lossless reference coding of existing mixed7/8 integers, without requantization."""
# Role: zlib helpers (compress; decompress with exact-length checks) and a pack format that stores
# a model with mixed 7-bit packed and int8 weights as residuals against member B's int8 codes.
# Used by: fd_entropy_codec (compress, decompress); not used by the delivered run (it is imported
# through fd_lzma_codec -> fd_entropy_codec, but none of its functions runs).
import zlib
from pathlib import Path
import numpy as np
import torch

from cuhkx.budget import _bit_pack,_bit_unpack
from cuhkx.fd_codec import tensor_spec,state_spec,check_state

# Format tag of this module's pack format; not the delivered checkpoint's.
FORMAT='cuhkx-fd-mixed-reference-v1'


def compress(a):
    # zlib level 6 of the array's raw bytes, returned as a flat uint8 tensor.
    return torch.from_numpy(np.frombuffer(zlib.compress(a.tobytes(),6),dtype=np.uint8).copy())


def decompress(blob,n):
    # Inflate at most n + 1 bytes; require exactly n bytes, the end of the stream, no trailing bytes
    # and no input held back by the output limit.
    if not isinstance(n,int) or n<0 or blob.dtype!=torch.uint8 or blob.ndim!=1:raise ValueError('Invalid residual schema')
    stream=zlib.decompressobj();raw=stream.decompress(blob.numpy().tobytes(),n+1)
    if len(raw)!=n or not stream.eof or stream.unused_data or stream.unconsumed_tail:raise ValueError('Truncated, oversized or trailing residual')
    return np.frombuffer(raw,dtype=np.uint8)


def prediction(base,bits):
    # This predicts an integer code for compression only. The residual always restores
    # the ORIGINAL target integer; its original scale is stored independently.
    # Rescale int8 codes (range -127..127) to the -qmax..qmax range of a `bits`-bit code (63 for 7).
    qmax=2**(bits-1)-1
    return np.rint(base.astype(np.float64)*qmax/127).astype(np.int16)


def encode(base,target):
    # Encode the target model against B entry by entry: 7-bit packed, int8 or verbatim (below).
    if set(base)!=set(target):raise ValueError('Different architecture/state keys')
    result={}
    for key,value in target.items():
        reference=base[key]
        if isinstance(value,dict) and '_packed' in value:
            # 7-bit packed target against B's int8 entry of the same shape: signed codes minus B's
            # codes rescaled to -63..63, mod 256, ordered by a stable sort of B's codes, zlib.
            if set(value)!={'_packed','_scale','_bits','_shape'} or not isinstance(reference,dict) or set(reference)!={'_int8','_scale'}:raise ValueError('Unsupported mixed/base schema')
            bits=int(value['_bits']);shape=tuple(map(int,value['_shape']));n=int(np.prod(shape))
            b=reference['_int8'];assert b.dtype==torch.int8 and tuple(b.shape)==shape and bits==7
            # Unpack and shift back to signed codes in -63..63.
            q=_bit_unpack(value['_packed'].numpy(),n,bits).astype(np.int16)-(2**(bits-1)-1)
            original=b.numpy().reshape(-1);pred=prediction(original,bits)
            residual=((q-pred)%256).astype(np.uint8)
            order=np.argsort(original,kind='stable')
            result[key]=dict(kind='mixed7-context-residual',residual=compress(residual[order]),elements=n,
                base_spec=tensor_spec(b),original_metadata={k:v for k,v in value.items() if k!='_packed'},target_spec=state_spec(value))
        elif isinstance(value,dict) and '_int8' in value:
            # int8 target: (target codes - B codes) mod 256, zlib; the target's scale is kept as is.
            if set(value)!={'_int8','_scale'} or not isinstance(reference,dict) or set(reference)!={'_int8','_scale'}:raise ValueError('Unsupported int8 schema')
            b,q=reference['_int8'],value['_int8']
            if b.dtype!=torch.int8 or q.dtype!=torch.int8 or b.shape!=q.shape:raise ValueError('Int8 shape/dtype mismatch')
            residual=((q.numpy().astype(np.int16)-b.numpy().astype(np.int16))%256).astype(np.uint8)
            result[key]=dict(kind='int8-residual',residual=compress(residual),elements=b.numel(),base_spec=tensor_spec(b),scale=value['_scale'],target_spec=state_spec(value))
        else:result[key]=dict(kind='verbatim',value=value,target_spec=state_spec(value))
    return result


def restore_primary(base,encoded):
    # Inverse of encode: check B's codes against the reference manifest, rebuild each entry and
    # check it against the target manifest.
    if set(base)!=set(encoded):raise ValueError('Reference keys differ')
    result={}
    for key,entry in encoded.items():
        kind=entry['kind']
        if kind=='verbatim':
            if set(entry)!={'kind','value','target_spec'}:raise ValueError('Invalid verbatim schema')
            value=entry['value']
        else:
            # Expected keys depend on the kind: 7-bit entries carry the original metadata (scale,
            # bits, shape), int8 entries the scale.
            expected={'kind','residual','elements','base_spec','target_spec'}|({'original_metadata'} if kind=='mixed7-context-residual' else {'scale'} if kind=='int8-residual' else set())
            if kind not in ('mixed7-context-residual','int8-residual') or set(entry)!=expected:raise ValueError('Invalid residual schema')
            b=base[key]['_int8']
            if tensor_spec(b)!=entry['base_spec'] or entry['elements']!=b.numel():raise ValueError('Reference SHA/elements mismatch')
            decoded=decompress(entry['residual'],entry['elements'])
            if kind=='int8-residual':
                # (B codes + residual) mod 256 read as int8 is the target's code.
                q=((b.numpy().astype(np.int16)+decoded.reshape(b.shape).astype(np.int16))%256).astype(np.uint8).view(np.int8)
                value={'_int8':torch.from_numpy(q.copy()),'_scale':entry['scale']}
            else:
                meta=entry['original_metadata']
                if set(meta)!={'_scale','_bits','_shape'} or int(meta['_bits'])!=7 or tuple(map(int,meta['_shape']))!=tuple(b.shape):raise ValueError('Mixed shape/bits mismatch')
                # Undo the ordering, add the prediction back modulo 256, check the -63..63 range and
                # re-pack at 7 bits after shifting the codes to 0..126.
                original=b.numpy().reshape(-1);order=np.argsort(original,kind='stable');residual=np.empty(len(original),dtype=np.uint8);residual[order]=decoded
                q=((prediction(original,7)+residual.astype(np.int16))%256).astype(np.uint8).view(np.int8)
                if (q<-63).any() or (q>63).any():raise ValueError('Invalid recovered int7 code')
                packed=_bit_pack((q.astype(np.int16)+63).astype(np.uint8),7)
                value=dict(_packed=torch.from_numpy(packed.copy()),**meta)
        check_state(value,entry['target_spec']);result[key]=value
    return result


def assemble(base,primary,thermal,detector,meta):
    # Pack of B, a 'thermal' state and the detector as they are and a 'primary' model stored
    # against B, with a manifest of each.
    return dict(format=FORMAT,version=1,meta=meta,base=base,primary_delta=encode(base,primary),thermal=thermal,detector=detector,
        tensor_manifest={k:state_spec(v) for k,v in dict(B=base,primary=primary,thermal=thermal,detector=detector).items()})


def restore(pack):
    if set(pack)!={'format','version','meta','base','primary_delta','thermal','detector','tensor_manifest'} or pack['format']!=FORMAT or pack['version']!=1:raise ValueError('Invalid mixed reference checkpoint')
    m=pack['tensor_manifest']
    if set(m)!={'B','primary','thermal','detector'}:raise ValueError('Invalid tensor manifest')
    # Check B, the 'thermal' state and the detector, then rebuild and check the primary model.
    for name,key in [('B','base'),('thermal','thermal'),('detector','detector')]:check_state(pack[key],m[name])
    primary=restore_primary(pack['base'],pack['primary_delta']);check_state(primary,m['primary'])
    return dict(B=pack['base'],primary=primary,thermal=pack['thermal']),pack['detector']


def load(path):
    with Path(path).open('rb') as f:pack=torch.load(f,map_location='cpu',weights_only=True)
    members,detector=restore(pack);return pack,members,detector
