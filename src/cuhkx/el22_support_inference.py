"""EL22 P1: unchanged F0 and encoder, final support-weighted pooling only."""
# Role: recognition stage of the delivered model: members B and C give F0; the frozen ViT-S
#   encodes det248 for IR and depth, its tokens are pooled over five regions with in-frame
#   support weights (P1), the readout head predicts q, and compose() blends q with F0.
# Used by: scripts/el25r_p1_repeat_runtime.py (started by inference.sh); inference.
import types,time
from pathlib import Path
import numpy as np
import torch
from cuhkx.fd18_codec import load
from cuhkx.fd18_readout import predict,compose
from cuhkx.fd15_vit import make,dequantize,preprocess
from cuhkx.el22_support_pool import extract_pair
from cuhkx.el22_runtime_support import support_for_ids
from cuhkx.fused import load_fused,tsn_picks

# Returns (base, readouts, features, telemetry). `specs` maps the views det, miw and det248 to
# their cache path and build record; `availability` maps each clip ID to 'I' and 'D' flags from
# the raw-frame decode check; `requested_ids` optionally restricts and orders the clips.
def components(pack_path,specs,availability,requested_ids=None):
    from cuhkx import fd13_inference as old
    # Step 1: read the checkpoint and verify its SHA256 manifests, giving the unpacked dict, the
    # int8 ViT encoder state, the readout heads, the member states (B, C) and the detector state.
    blob,eq,heads,bm,det=load(pack_path)
    # Stand-in for fd13_codec.load inside infer_f0 (same checkpoint only): returns the member
    # base already unpacked and verified above instead of reading the file again.
    def base_load(path):assert Path(path).resolve()==Path(pack_path).resolve();return blob['base'],bm,det
    # Step 2, the statements of the next line in order:
    # 1) basefn is a copy of fd13_inference.infer_f0 whose global `load` is base_load (the
    #    module itself is unchanged). It runs members B and C on the det and miw caches with
    #    4-pass TTA and no diagnostic masking ('none'), giving the components `base` and the
    #    telemetry `t`.
    # 2) F0 (base['fused'] = 1/2 B + 1/2 C) is also kept as 'F0'; `names` is the clip order used
    #    from here on.
    # 3) The det248 cache is opened (memory-mapped uint8 frames, per-clip offsets, clip IDs);
    #    `rows` is each clip's row in it, -1 if the clip is missing.
    # 4) `weights`: fraction of each 16x16 patch inside the camera frame, shape
    #    (clips, 2 views, 14, 14); the full-data checkpoint (fold -1) reads the test split's raw
    #    frames and person windows, a fold checkpoint the training split's.
    # 5) The ViT-S encoder is built on the GPU in eval mode and its tensors are replaced by the
    #    dequantised int8 encoder; strict loading fails on any key the model lacks. The 40-class
    #    head of make() is not stored in the checkpoint and is not used.
    # 6) torch uses one CPU thread.
    basefn=types.FunctionType(old.infer_f0.__code__,dict(old.infer_f0.__globals__,load=base_load),'infer_f0',old.infer_f0.__defaults__);base,t=basefn(pack_path,{k:specs[k] for k in ['det','miw']},requested_ids,'none');base['F0']=base['fused'].copy();names=base['clip_ids'];data,off,ids,_=load_fused(Path(specs['det248']['path']));lookup={str(c):i for i,c in enumerate(ids)};rows=np.array([lookup.get(str(c),-1) for c in names]);weights=support_for_ids(list(map(str,names)), 'train' if blob['meta']['fold']>=0 else 'test');m=make().cuda().eval();state=m.state_dict();state.update(dequantize(eq));m.load_state_dict(state,strict=True);features={};torch.set_num_threads(1)
    # Step 3: for each modality (I: IR copied to three channels; D: Depth_Color RGB), encode 16
    # frames of det248 in the plain and flipped views and keep the support-weighted region
    # features (P1).
    for mod in ['I','D']:
        # A clip is present for this modality if it has frames in the det248 cache and
        # `availability` marks the modality usable; kept as base['I_present'] and
        # base['D_present'] for compose(). `regions` is (clips, 2 views, 5 regions, 384), zero
        # until filled.
        present=np.array([r>=0 and off[r+1]>off[r] and availability[str(c)][mod] for c,r in zip(names,rows)],bool);base[mod+'_present']=present;regions=np.zeros((len(names),2,5,384),np.float32);start=time.perf_counter()
        if present.any():
            # Missing clips borrow the first present clip's cache row and support weights, so the
            # batches keep the size and grouping they have with every clip present (as in
            # fd13_inference.infer_f0). Their features are not used by the fusion: compose()
            # applies the readout only to clips with both modalities present.
            use=rows.copy();use[~present]=rows[present][0];use_weights=weights.copy();use_weights[~present]=weights[present][0]
            # No autograd; cuDNN without autotuning, without TF32 and outside its deterministic
            # mode (the previous cuDNN flags are restored when the block ends).
            with torch.no_grad(),torch.backends.cudnn.flags(enabled=True,benchmark=False,deterministic=False,allow_tf32=False):
                # Full FP32 matrix multiplies as well; this global setting stays after the block.
                torch.backends.cuda.matmul.allow_tf32=False
                # Batches of 8 clips.
                for begin in range(0,len(use),8):
                    # For each clip of the batch: its stored frames (at most 32), 16 TSN picks (the
                    # centre frame of each of 16 equal segments; short clips repeat frames), as
                    # (T, C, H, W); stacked into a (batch, 16, 4, 248, 248) uint8 GPU tensor.
                    rs=use[begin:begin+8];raw=torch.from_numpy(np.stack([np.ascontiguousarray(data[off[r]:off[r+1]][tsn_picks(off[r+1]-off[r],16,None)].transpose(0,3,1,2)) for r in rs])).cuda()
                    # view 0 = plain, 1 = horizontal flip (the weights were computed for the same
                    # flip). preprocess(): modality channels, centre 224 crop, ImageNet
                    # normalisation, flip. extract_pair() returns the plain (P0) and the
                    # support-weighted (P1) region features, (batch, 5, 384) each; P1 is kept.
                    for view in [0,1]:
                        _,weighted=extract_pair(m,preprocess(raw,mod,bool(view)),torch.from_numpy(use_weights[begin:begin+len(rs),view]).cuda())
                        regions[begin:begin+len(rs),view]=weighted.cpu().numpy()
        # 'SI'/'SD': the five-region features, input of the S1 head. 'I'/'D': the whole-grid
        # region only, (clips, 2, 384), for head families on global features (not S1). Plus the
        # encoding time of this modality.
        features['S'+mod]=regions;features[mod]=regions[:,:,0];t[mod+'_region_seconds']=time.perf_counter()-start
    # Temporal head family T0: skipped for the delivered checkpoint, whose only head is S1 (the
    # module cuhkx.fd18_temporal it would import is not part of this package).
    if 'T0' in heads:
        from cuhkx.fd18_temporal import extract_raw
        temporal,receipt=extract_raw(m,list(map(str,names)),split='train' if blob['meta']['fold']>=0 else 'test',phases=[.25,.75])
        features.update({'T0-'+k:v for k,v in temporal.items()});t['raw_temporal']=receipt
    # Step 4: readout prediction. Each head gives (clips, 40) class probabilities averaged over
    # the plain and flipped views (fd18_readout.predict); T0 would read its temporal features
    # in place of 'I' and 'D'.
    readouts={}
    for family,h in heads.items():readouts[family]=predict({**features,**{k:features['T0-'+k] for k in ['I','D']}} if family=='T0' else features,h)
    # Step 5: telemetry: peak GPU memory and two constant descriptive flags.
    t.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),single_public_encoder=True,regions_before_spatial_average=True)
    return base,readouts,features,t

# Runs components() once and blends F0 with the readout for each requested configuration name
# (e.g. 'S1-G'), with no modality hidden (compose's default phase 'warm'). Returns (results by
# name, base, readouts, features, telemetry).
def infer(pack_path,specs,availability,candidates,requested_ids=None):
    b,q,f,t=components(pack_path,specs,availability,requested_ids);return {c:compose(b,q,c) for c in candidates},b,q,f,t
