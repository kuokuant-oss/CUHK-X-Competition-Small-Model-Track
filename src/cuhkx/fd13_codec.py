"""FD13 generic optional-member storage. F0 tensors are losslessly preserved."""
# Role: the member base inside the checkpoint: member B, member C as an int8 delta against B, the
# person detector and optional extra members, with a per-tensor SHA256 manifest for each and a
# checksum over the metadata. restore() decodes and checks it; finish() and add_member() build it.
# Used by: fd18_codec.load (restore); inference. load() and add_member() are not called by the
# delivered run (el22_support_inference runs fd13_inference.infer_f0 on the member base that
# fd18_codec.load has already restored).
# The delivered pack has no optional members: METHOD.md §5 accounts for every part of
# weights/model.pt (B, C, ViT encoder, detector, readout head), and the member inference
# (fd13_inference.infer_f0) runs only B and C.
import hashlib
import json
from pathlib import Path
import torch
from cuhkx.fd_codec import check_state, state_spec
from cuhkx.fd_lzma_codec import restore_state, restore_delta, encode_state
# fd_lzma_codec's codec for 7-bit packed states (named 'thermal' there); only optional members use
# it here.
from cuhkx.fd_lzma_codec import encode_thermal as encode_unsigned7_state
from cuhkx.fd_lzma_codec import restore_thermal as restore_unsigned7_state

# Format tag that restore() requires.
FORMAT='cuhkx-fd13-optional-v1'

def metadata_hash(meta):
    # SHA256 of canonical JSON: sorted keys, no spaces, NaN and infinity rejected.
    return hashlib.sha256(json.dumps(meta,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def finish(pack):
    # Store the checksum over the metadata and the optional members' specs (checked by restore).
    pack['metadata_sha256']=metadata_hash(dict(meta=pack['meta'],optional_specs={k:v['spec'] for k,v in pack['optional'].items()}))
    return pack

def restore(pack):
    # Check: exact top-level keys, format tag and version 1.
    if set(pack)!={'format','version','meta','base','C_delta','detector','tensor_manifest','optional','metadata_sha256'} or pack['format']!=FORMAT or pack['version']!=1:raise ValueError('Invalid FD13 schema')
    meta=pack['meta'];optional=pack['optional'];manifest=pack['tensor_manifest']
    # Check: optional members may only be CSN, K3 or K6, and the manifest names exactly B, C, the
    # detector and those optional members.
    if not set(optional)<= {'CSN','K3','K6'} or set(manifest)!= {'B','C','detector',*optional}:raise ValueError('Invalid member names')
    # Check: the checksum over the metadata and the optional members' specs.
    if pack['metadata_sha256']!=metadata_hash(dict(meta=meta,optional_specs={k:v['spec'] for k,v in optional.items()})):raise ValueError('Metadata checksum mismatch')
    # Check: the fixed F0 recipe in the metadata: classes in order 0..39, a pipeline of exactly B
    # and C, weights 0.5/0.5 with power 1 (F0 = ½ B + ½ C) and joint (MAP) decoding.
    if meta['class_order']!=list(range(40)) or set(meta['pipeline'])!={'B','C'}:raise ValueError('Invalid F0 pipeline')
    if meta['fusion']['weights']!=[.5,.5] or meta['fusion']['power']!=1 or meta['decoder']['mode']!='MAP':raise ValueError('F0 contract changed')
    # Decode B's tensors, rebuild C from B and the stored residuals (fd_lzma_codec.restore_delta
    # checks C's SHA256 tensor by tensor) and decode the detector state (FP32, not quantised).
    base=restore_state(pack['base']);members=dict(B=base,C=restore_delta(base,pack['C_delta']));det=restore_state(pack['detector'])
    # Optional members (none in the delivered pack): check the spec (its own name, 4 input channels,
    # 32 frames; CSN must be ir-CSN-152 on 224² crops of det248), then decode the stored form:
    # 7-bit packed codes, an encoded state like B's, or a delta against member C.
    for name,e in optional.items():
        if set(e)!= {'spec','storage','data'}:raise ValueError('Invalid optional member schema')
        spec=e['spec']
        if spec['name']!=name or spec['in_channels']!=4 or spec['n_frames']!=32:raise ValueError('Invalid optional identity/input')
        if name=='CSN' and (spec['arch']!='ircsn152' or spec['crop']!=224 or spec['view']!='det248'):raise ValueError('CSN is four-channel det248/crop224')
        if e['storage']=='unsigned7':value=restore_unsigned7_state(e['data'])
        elif e['storage']=='state':value=restore_state(e['data'])
        elif e['storage']=='C-delta':value=restore_delta(members['C'],e['data'])
        else:raise ValueError('Unknown optional storage')
        members[name]=value
    # Final check: every member and the detector against the manifest (same keys; dtype, shape,
    # element count and SHA256 of every tensor).
    for k,v in dict(**members,detector=det).items():check_state(v,manifest[k])
    return members,det

def load(path):
    # Reads a file that holds a member base on its own. The delivered checkpoint embeds the member
    # base, and fd18_codec.load passes it to restore() directly.
    with Path(path).open('rb') as f:pack=torch.load(f,map_location='cpu',weights_only=True)
    members,det=restore(pack)
    return pack,members,det

def add_member(pack,name,state,spec,storage='unsigned7'):
    # Builder: add an optional member as 7-bit packed codes or as an encoded state, record its
    # manifest and refresh the metadata checksum. Nothing in this package calls it.
    if name in pack['optional']:raise ValueError('Duplicate optional member')
    if storage=='unsigned7':data=encode_unsigned7_state(state)
    elif storage=='state':data=encode_state(state)
    else:raise ValueError('Unsupported builder storage')
    pack['optional'][name]=dict(spec=spec,storage=storage,data=data)
    pack['tensor_manifest'][name]=state_spec(state)
    return finish(pack)
