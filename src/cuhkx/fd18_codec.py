"""Single-file lossless envelope with strict byte length and SHA verification."""
# Role: writes and reads weights/model.pt. The file is a zlib envelope around one serialised pack
# that holds the member base (B, C as an int8 delta against B, detector), the int8 ViT encoder,
# the readout heads and the metadata, each covered by a SHA256 manifest or checksum.
# Used by: the inference entry point scripts/el25r_p1_repeat_runtime.py (load, unpack),
# el22_support_inference.components (load) and tools/make_preprocessing_figure.py (load);
# inference. encode() writes the format and is not called at inference.
import io,zlib,hashlib,json
import numpy as np
import torch
# state_spec/check_state are fd16_codec's general manifest functions: besides tensors they cover
# lists and plain values (strings, numbers), which the member base, heads and fusion descriptions
# contain.
from cuhkx.fd16_codec import encode as base_encode,state_spec,check_state
# restore_base decodes the member base: B, C rebuilt from its delta, and the detector.
from cuhkx.fd13_codec import restore as restore_base
# graph(c) returns the fusion description of readout c (family, blend kind, weight 0.2, ...).
from cuhkx.fd18_readout import graph

def encode(base,encoder,heads,meta):
    # Inner pack: fd16_codec's layout (member base, zlib-compressed encoder and heads with their
    # manifests, metadata with its SHA256) under this format's tag, plus a manifest of the encoded
    # member base and a fusion description (with its own manifest) per readout in meta['selected'].
    p=base_encode(base,encoder,heads,meta);p['format']='cuhkx-fd18-v1';p['base_manifest']=state_spec(base);p['member_graph']={c:graph(c) for c in meta['selected']};p['graph_manifest']=state_spec(p['member_graph'])
    # Serialise the inner pack with torch.save and compress the bytes with zlib level 6.
    stream=io.BytesIO();torch.save(p,stream);raw=stream.getvalue();compressed=zlib.compress(raw,6)
    # The envelope records the codec and the exact length and SHA256 of the uncompressed bytes; the
    # compressed bytes are stored as a flat uint8 tensor.
    return dict(format='cuhkx-fd18-envelope-v1',codec='zlib6',raw_length=len(raw),raw_sha256=hashlib.sha256(raw).hexdigest(),payload=torch.from_numpy(np.frombuffer(compressed,dtype=np.uint8).copy()))

def unpack(envelope,loader=None):
    # `envelope` is what torch.load returns for weights/model.pt. `loader` replaces torch.load for
    # the inner pack; the entry point's detector-only step, which patches torch.load, passes the
    # function it replaced.
    # Check: the envelope's format tag and codec name, and the payload is a flat uint8 tensor.
    assert envelope['format']=='cuhkx-fd18-envelope-v1' and envelope['codec']=='zlib6';assert envelope['payload'].dtype==torch.uint8 and envelope['payload'].ndim==1
    # Check: the recorded length is positive and below 150,000,000 bytes, which bounds the buffer.
    length=envelope['raw_length'];assert 0<length<150000000
    # Check: inflate at most length + 1 bytes, then require the end of the zlib stream (eof), no
    # bytes after it (unused_data), no input held back by the output limit (unconsumed_tail) and
    # exactly `length` bytes out; last, the SHA256 of those bytes must equal the recorded one.
    dz=zlib.decompressobj();raw=dz.decompress(envelope['payload'].numpy().tobytes(),length+1);assert dz.eof and not dz.unused_data and not dz.unconsumed_tail and len(raw)==length;assert hashlib.sha256(raw).hexdigest()==envelope['raw_sha256']
    # Deserialise the inner pack with the weights-only unpickler (tensors and plain containers).
    return (loader or torch.load)(io.BytesIO(raw),map_location='cpu',weights_only=True)

def load(path):
    # Checks, in order: the envelope (unpack), the inner format tag, the metadata SHA256 (JSON with
    # sorted keys), the fusion descriptions against their manifest, and their equality with
    # fd18_readout.graph() for every selected readout, so the stored description matches the code.
    p=unpack(torch.load(path,map_location='cpu',weights_only=True));assert p['format']=='cuhkx-fd18-v1';assert p['meta_sha256']==hashlib.sha256(json.dumps(p['meta'],sort_keys=True).encode()).hexdigest();check_state(p['member_graph'],p['graph_manifest']);assert p['member_graph']=={c:graph(c) for c in p['meta']['selected']}
    # Inflate and deserialise the int8 ViT encoder state (fd15_vit.dequantize turns it into FP32
    # weights at inference). Its compressed bytes are already covered by the envelope SHA256.
    e=torch.load(io.BytesIO(zlib.decompress(p['encoder'].numpy().tobytes())),map_location='cpu',weights_only=True)
    # Check: the encoded member base, the decoded encoder state and the readout heads each equal
    # their manifest (dtype, shape, element count and SHA256 of every tensor; plain values exactly).
    for v,spec in [(p['base'],p['base_manifest']),(e,p['encoder_manifest']),(p['heads'],p['heads_manifest'])]:check_state(v,spec)
    # Decode the member base; fd13_codec.restore checks B, the rebuilt C and the detector against
    # its per-tensor manifest. Returns the inner pack, encoder state, heads, members and detector.
    members,det=restore_base(p['base']);return p,e,p['heads'],members,det
