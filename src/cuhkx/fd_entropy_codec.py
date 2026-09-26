"""Exact scale-predicted int8 residuals and unpacked-code entropy coding."""
# Role: prediction(), which predicts one int8 tensor's codes from a reference int8 tensor and the
# two per-output-channel scales; the rest is a zlib-only pack format (scale-context int8
# residuals, unpacked 7-bit codes) that the delivered checkpoint does not use.
# Used by: fd_lzma_codec (prediction, for the 3 tensors of member C stored as
# 'scale-context-residual', METHOD.md §5); inference. The other functions are not used by the
# delivered run.
from pathlib import Path
import numpy as np
import torch

from cuhkx.budget import _bit_pack, _bit_unpack
from cuhkx.fd_codec import tensor_spec, state_spec, check_state
from cuhkx.fd_mixed_codec import compress, decompress

# Format tag of this module's zlib-only pack format; not the delivered checkpoint's.
FORMAT = 'cuhkx-fd-entropy-v1'


def prediction(base, scale):
    # Predict a target's int8 codes from the reference pair `base` ({'_int8', '_scale'}) and the
    # target's scale: round(codes x reference scale / target scale), clipped to -127..127, as
    # int16, i.e. the reference weights expressed in the target's quantisation step.
    q = base['_int8']
    if q.dtype != torch.int8 or q.ndim < 2:
        raise ValueError('Invalid reference integer tensor')
    bscale = base['_scale'].numpy().astype(np.float64)
    tscale = scale.numpy().astype(np.float64)
    # Both scales must be per output channel, shape (out_channels, 1, ..., 1), and finite.
    expected = (q.shape[0],) + (1,) * (q.ndim - 1)
    if bscale.shape != expected or tscale.shape != expected or not np.isfinite(bscale).all() or not np.isfinite(tscale).all():
        raise ValueError('Invalid original scale schema')
    # A target scale can be 0 (the FP16 scale of an all-zero channel); there the ratio stays 1.
    ratio = np.ones_like(tscale)
    np.divide(bscale, tscale, out=ratio, where=tscale != 0)
    # A compression predictor only; the residual recovers the original target q.
    return np.clip(np.rint(q.numpy().astype(np.float64) * ratio), -127, 127).astype(np.int16)


# The functions below form this module's zlib-only pack format, which the delivered checkpoint
# does not use; its member C is stored by fd_lzma_codec.encode_delta, which offers this residual
# as one of two modes and also tries LZMA.
def encode_delta(base, target):
    # Every int8 pair becomes (target codes - prediction) mod 256, reordered by a stable sort of
    # the reference codes and compressed with zlib; other entries are kept verbatim.
    if set(base) != set(target):
        raise ValueError('Architecture/state keys differ')
    out = {}
    for key, value in target.items():
        reference = base[key]
        if isinstance(value, dict) and set(value) == {'_int8', '_scale'}:
            if not isinstance(reference, dict) or set(reference) != {'_int8', '_scale'}:
                raise ValueError('Invalid int8 reference')
            b, q = reference['_int8'], value['_int8']
            if q.dtype != torch.int8 or q.shape != b.shape:
                raise ValueError('Target dtype/shape mismatch')
            residual = ((q.numpy().astype(np.int16) - prediction(reference, value['_scale'])) % 256).astype(np.uint8)
            # Stable context ordering is wholly determined by the reference.
            order = np.argsort(b.numpy().reshape(-1), kind='stable')
            out[key] = dict(kind='scale-context-int8', residual=compress(residual.reshape(-1)[order]), elements=q.numel(),
                base_spec=state_spec(reference), scale=value['_scale'], target_spec=state_spec(value))
        else:
            out[key] = dict(kind='verbatim', value=value, target_spec=state_spec(value))
    return out


def restore_delta(base, encoded):
    # Inverse of encode_delta: check the reference, undo the reordering, add the prediction back
    # modulo 256 and read the bytes as int8; each entry is checked against the target manifest.
    if set(base) != set(encoded):
        raise ValueError('Reference keys differ')
    out = {}
    for key, entry in encoded.items():
        if entry['kind'] == 'verbatim':
            if set(entry) != {'kind', 'value', 'target_spec'}:
                raise ValueError('Invalid verbatim schema')
            value = entry['value']
        elif entry['kind'] == 'scale-context-int8':
            if set(entry) != {'kind', 'residual', 'elements', 'base_spec', 'scale', 'target_spec'}:
                raise ValueError('Invalid residual schema')
            check_state(base[key], entry['base_spec'])
            b = base[key]['_int8']
            if entry['elements'] != b.numel():
                raise ValueError('Invalid element count')
            decoded = decompress(entry['residual'], entry['elements'])
            order = np.argsort(b.numpy().reshape(-1), kind='stable')
            residual = np.empty(b.numel(), dtype=np.uint8)
            residual[order] = decoded
            q = ((prediction(base[key], entry['scale']) + residual.reshape(b.shape).astype(np.int16)) % 256).astype(np.uint8).view(np.int8)
            value = dict(_int8=torch.from_numpy(q.copy()), _scale=entry['scale'])
        else:
            raise ValueError('Unknown residual kind')
        check_state(value, entry['target_spec'])
        out[key] = value
    return out


def encode_packed(target):
    # 7-bit packed entries -> the unpacked codes (one byte each) compressed with zlib, plus the
    # unused high bits of the last packed byte; the same scheme as fd_lzma_codec.encode_thermal.
    out = {}
    for key, value in target.items():
        if isinstance(value, dict) and '_packed' in value:
            if set(value) != {'_packed', '_scale', '_bits', '_shape'}:
                raise ValueError('Invalid packed schema')
            bits = int(value['_bits']); shape = tuple(map(int, value['_shape'])); n = int(np.prod(shape))
            packed = value['_packed']
            if bits != 7 or not shape or min(shape) <= 0 or packed.dtype != torch.uint8 or packed.ndim != 1 or packed.numel() != (n * bits + 7) // 8:
                raise ValueError('Invalid packed tensor')
            codes = _bit_unpack(packed.numpy(), n, bits)
            used = n * bits % 8
            padding = int(packed[-1]) & (255 ^ ((1 << used) - 1)) if used else 0
            out[key] = dict(kind='unpacked7-zlib', codes=compress(codes), elements=n, padding=padding,
                original_metadata={k:v for k,v in value.items() if k != '_packed'}, target_spec=state_spec(value))
        else:
            out[key] = dict(kind='verbatim', value=value, target_spec=state_spec(value))
    return out


def restore_packed(encoded):
    # Inverse of encode_packed; the re-packed tensor is checked against the original manifest.
    out = {}
    for key, entry in encoded.items():
        if entry['kind'] == 'verbatim':
            if set(entry) != {'kind', 'value', 'target_spec'}:
                raise ValueError('Invalid verbatim schema')
            value = entry['value']
        elif entry['kind'] == 'unpacked7-zlib':
            if set(entry) != {'kind', 'codes', 'elements', 'padding', 'original_metadata', 'target_spec'}:
                raise ValueError('Invalid unpacked schema')
            meta = entry['original_metadata']
            if set(meta) != {'_scale', '_bits', '_shape'} or int(meta['_bits']) != 7:
                raise ValueError('Invalid packed metadata')
            shape = tuple(map(int, meta['_shape'])); n = entry['elements']
            if not shape or min(shape) <= 0 or n != int(np.prod(shape)):
                raise ValueError('Invalid packed shape')
            codes = decompress(entry['codes'], n)
            if (codes > 127).any():
                raise ValueError('Invalid original unsigned7 codes')
            used = n * 7 % 8; mask = 255 ^ ((1 << used) - 1) if used else 0
            padding = entry['padding']
            if not isinstance(padding, int) or padding < 0 or padding > 255 or padding & ~mask:
                raise ValueError('Invalid packed padding')
            packed = _bit_pack(codes, 7)
            if used: packed[-1] |= padding
            value = dict(_packed=torch.from_numpy(packed.copy()), **meta)
        else:
            raise ValueError('Unknown packed kind')
        check_state(value, entry['target_spec']); out[key] = value
    return out


def assemble(base, target, thermal, detector, meta):
    # Pack of B and the detector as they are, C as scale-context residuals and a 7-bit packed state
    # as unpacked codes, with a manifest of each.
    return dict(format=FORMAT, version=1, meta=meta, base=base, C_delta=encode_delta(base, target), thermal_codes=encode_packed(thermal), detector=detector,
        tensor_manifest={k:state_spec(v) for k,v in dict(B=base, C=target, thermal=thermal, detector=detector).items()})


def restore(pack):
    if set(pack) != {'format', 'version', 'meta', 'base', 'C_delta', 'thermal_codes', 'detector', 'tensor_manifest'} or pack['format'] != FORMAT or pack['version'] != 1:
        raise ValueError('Invalid entropy checkpoint')
    m = pack['tensor_manifest']
    if set(m) != {'B', 'C', 'thermal', 'detector'}:
        raise ValueError('Invalid manifest')
    # Check B and the detector, rebuild C and the packed state, then check every member.
    check_state(pack['base'], m['B']); check_state(pack['detector'], m['detector'])
    members = dict(B=pack['base'], C=restore_delta(pack['base'], pack['C_delta']), thermal=restore_packed(pack['thermal_codes']))
    for k,v in members.items(): check_state(v, m[k])
    return members, pack['detector']


def load(path):
    with Path(path).open('rb') as f: pack = torch.load(f, map_location='cpu', weights_only=True)
    members, detector = restore(pack)
    return pack, members, detector
