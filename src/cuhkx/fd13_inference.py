"""FD member inference from one checkpoint, with original per-member pipelines."""
# Role: runs members B and C (R(2+1)D-34 with a 4-channel stem) on the det and miw caches with the
#   4-pass TTA and returns their probabilities and F0 = 1/2 B + 1/2 C.
# Used by: el22_support_inference.components, which runs a copy of infer_f0 on the member base
#   inside weights/model.pt; inference. infer_caches is not used by the delivered run.
import json
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

from cuhkx.fd_codec import fp32
from cuhkx.fd13_codec import load
from cuhkx.fused import FusedFrameDataset,load_fused,make_fused_transform
from cuhkx.models import Classifier,PretrainedVideo3D
from cuhkx.thermal_pair import fixed_pair
from cuhkx.train import seed_everything
from cuhkx.tta import predict


# Builds one member network (Classifier on PretrainedVideo3D) from its pipeline spec and stored
# int8 state, without network access or pretrained downloads.
def model_from_spec(spec,state):
    # Seed 42 and deterministic GPU kernels (train.seed_everything).
    seed_everything(42,deterministic=True)
    # R(2+1)D-34 architecture code bundled under runtime/ (from ig65m-pytorch).
    hub=Path(__file__).resolve().parents[2]/'runtime/torch/hub/moabitcoin_ig65m-pytorch_master'
    if not (hub/'hubconf.py').is_file():raise FileNotFoundError('Bundled architecture source is required')
    # While the network is built, torch.hub.load is redirected to the bundled copy; any other
    # repository, or a request for pretrained weights, is refused. The original is restored after.
    original=torch.hub.load
    def local_load(repo,name,*args,**kwargs):
        if repo!='moabitcoin/ig65m-pytorch' or kwargs.get('pretrained') is not False:raise ValueError('No remote/pretrained model loading')
        kwargs.pop('trust_repo',None)
        return original(str(hub),name,*args,source='local',**kwargs)
    torch.hub.load=local_load
    try:
        net=Classifier(PretrainedVideo3D(arch=spec['arch'],in_channels=spec['in_channels'],dropout=spec['dropout'],pretrained=False))
    finally:torch.hub.load=original
    # fp32() rebuilds FP32 weights from the stored int8 codes and per-channel scales.
    net.load_state_dict(fp32(state));return net


# Runs members B and C. `cache_specs` maps the views det and miw to their cache path and expected
# build record; `missing_local` is a diagnostic that hides the miw view on no clip ('none'),
# every 7th clip ('partial') or all clips ('all'). Returns (components, telemetry); components
# hold B, C_det and C_miw with their clip IDs, C, F0 as 'fused', and local_present.
def infer_f0(pack_path,cache_specs,requested_ids=None,missing_local='none'):
    if missing_local not in ('none','partial','all'):raise ValueError('Invalid local diagnostic')
    # load() reads the member base; in the delivered run el22_support_inference replaces it with
    # the base already unpacked from weights/model.pt. The asserts pin the F0 recipe: classes
    # 0-39, weights 0.5/0.5 with power 1, members B and C, and C_det as the miw fallback.
    packed,members,detector=load(pack_path);meta=packed['meta']
    assert meta['class_order']==list(range(40)) and meta['fusion']['weights']==[.5,.5] and meta['fusion']['power']==1
    assert {'B','C'}<=set(members) and meta['local_fallback']=='C_det; ensemble remains .5 B+.5 C_det'
    assert torch.cuda.is_available()
    # scripts/26_build_test_caches.py (scripts/ is on sys.path) provides compare_builds().
    torch.cuda.reset_peak_memory_stats(); builder=import_module('26_build_test_caches')
    caches={};names=None;metadata={}
    # Step 1: open each view's cache and line up its clips.
    for view,spec in cache_specs.items():
        # Memory-mapped (frames, 144, 144, 4) uint8 cache; its recorded build must agree with the
        # build the checkpoint expects.
        cache=Path(spec['path']);data,offsets,ids,build=load_fused(cache)
        assert isinstance(data,np.memmap) and data.shape[-3:]==(144,144,4)
        clashes,_=builder.compare_builds(spec['build'],list(build));assert not clashes,clashes
        ids=list(map(str,ids));assert len(ids)==len(set(ids))
        # All clips of the cache, or the requested ones; every view must give the same list.
        chosen=ids if requested_ids is None else list(map(str,requested_ids))
        lookup={c:i for i,c in enumerate(ids)};assert set(chosen)<=set(lookup)
        if names is None:names=chosen
        else:assert names==chosen
        # Cache row of each clip, and whether the clip stored any frame in this view.
        order=np.array([lookup[c] for c in names]);present=offsets[order+1]>offsets[order]
        # det must cover every clip; the diagnostics can hide only miw frames.
        if view=='det':assert present.all()
        elif missing_local=='all':present[:]=False
        elif missing_local=='partial':present[np.arange(len(present))%7==0]=False
        caches[view]=(data,offsets,order,present)
        metadata[view]=dict(path=str(cache),build=list(build),present=int(present.sum()),rows=len(data))
    # Step 2: each member on each of its views (B: det; C: det and miw) with the 4-pass TTA, at
    # the batch size stored in the checkpoint (8 for B and 6 for C in weights/model.pt).
    result={};times={};start=time.monotonic()
    for member in ('B','C'):
        spec=meta['pipeline'][member];net=model_from_spec(spec,members[member]).to('cuda').eval()
        for view in spec['views']:
            data,offsets,order,present=caches[view];rows=order[present];t=time.monotonic()
            # Keep the original batch shapes for surviving local observations. Dropping
            # absent rows changes the final CUDA batch and can change fp32 probabilities.
            # A real present observation fills unused slots; its outputs are discarded.
            # Only the miw view can be partly missing, since det covers every clip.
            keep_slots=view=='miw' and present.any() and not present.all()
            if keep_slots:
                rows=order.copy();rows[~present]=order[present][0]
            if len(rows):
                # Evaluation dataset over the selected rows: dummy labels, 32 TSN centre frames,
                # centre 128 crop, no augmentation.
                ds=FusedFrameDataset(data,offsets,np.zeros(len(offsets)-1,dtype=np.int64),rows,32,False,make_fused_transform(128,erase_prob=0,flip_prob=0),42)
                probs=predict(net,ds,'cuda',spec['batch_size'],True);del ds
                if keep_slots:probs=probs[present]
            else:probs=np.empty((0,40),dtype=np.float32)
            # Stored as 'B', 'C_det' or 'C_miw', with the IDs of the clips covered; one JSON
            # progress line per member and view.
            key=member if member=='B' else 'C_'+view
            result[key]=probs;result[key+'_ids']=np.asarray(names)[present];times[key]=time.monotonic()-t
            print(json.dumps(dict(member=member,view=view,clips=len(rows),batch=spec['batch_size'],seconds=times[key])),flush=True)
        del net;torch.cuda.empty_cache()
    # Step 3: C = mean of C_det and C_miw where miw is present, else C_det
    # (thermal_pair.fixed_pair); F0 = 1/2 B + 1/2 C in FP64. The assert checks that clips
    # without miw get exactly 1/2 B + 1/2 C_det.
    c,present=fixed_pair(names,result['C_det'],result['C_miw_ids'],result['C_miw'])
    fused=.5*(result['B'].astype(np.float64)+c.astype(np.float64))
    assert np.array_equal(fused[~present],.5*(result['B'][~present].astype(np.float64)+result['C_det'][~present].astype(np.float64)))
    result.update(clip_ids=np.asarray(names),C=c,fused=fused,local_present=present)
    # Telemetry: seconds per member and view, peak GPU memory, cache metadata and batch sizes.
    return result,dict(member_seconds=times,wall_seconds=time.monotonic()-start,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),metadata=metadata,missing_local=missing_local,local_fallback_exact=True,original_batch_by_member={k:v['batch_size'] for k,v in meta['pipeline'].items()})


# Optional-member path: adds an ir-CSN-152 member on the det248 view (224 crop, 4-pass TTA) to F0
# as 0.8 F0 + 0.2 CSN. Not used by the delivered run: nothing in this package calls it, it reads
# the checkpoint with fd13_codec.load (which does not accept weights/model.pt), and the member
# base inside weights/model.pt has an empty 'optional' table (no CSN weights).
def infer_caches(pack_path,cache_specs,requested_ids=None,missing_local='none',missing_optional='none'):
    if missing_optional not in ('none','partial','all'):raise ValueError('Invalid optional diagnostic')
    result,telemetry=infer_f0(pack_path,{k:cache_specs[k] for k in ('det','miw')},requested_ids,missing_local)
    pack,members,_=load(pack_path);names=result['clip_ids'];parent=result['fused'].copy();result['F0']=parent
    if not pack['optional']:return result,telemetry
    if set(pack['optional'])!={'CSN'}:raise ValueError('This inference entry requires CSN optional member')
    spec=pack['optional']['CSN']['spec'];cs=cache_specs.get('det248');present=np.zeros(len(names),dtype=bool);q=np.empty((0,40),dtype=np.float32);t=time.monotonic()
    if cs is not None and Path(cs['path']).with_name(Path(cs['path']).stem+'_meta.npz').exists():
        data,off,ids,build=load_fused(Path(cs['path']));assert isinstance(data,np.memmap) and data.shape[-3:]==(248,248,4)
        clashes,_=import_module('26_build_test_caches').compare_builds(cs['build'],list(build));assert not clashes,clashes
        lookup={str(c):i for i,c in enumerate(ids)};order=np.asarray([lookup.get(str(c),-1) for c in names]);present=np.asarray([i>=0 and off[i+1]>off[i] for i in order])
        if missing_optional=='partial':present[np.arange(len(names))%7==0]=False
        elif missing_optional=='all':present[:]=False
        if present.any():
            rows=order.copy();rows[~present]=order[present][0]
            ds=FusedFrameDataset(data,off,np.zeros(len(off)-1,dtype=np.int64),rows,32,False,make_fused_transform(224,erase_prob=0,flip_prob=0),42)
            net=model_from_spec(spec,members['CSN']).cuda().eval();q=predict(net,ds,'cuda',spec['batch_size'],True)[present];del net,ds;torch.cuda.empty_cache()
    fused=parent.copy();fused[present]=.8*parent[present]+.2*q.astype(np.float64)
    result.update(CSN=q,CSN_ids=names[present],optional_present=present,fused=fused)
    telemetry.update(optional_seconds=time.monotonic()-t,missing_optional=missing_optional,optional_present=int(present.sum()),optional_batch=spec['batch_size'],optional_channels=4,optional_crop=224,optional_fallback_exact=bool(np.array_equal(fused[~present],parent[~present])),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    return result,telemetry


# The same function again: this second, identical definition replaces the first at import.
def infer_caches(pack_path,cache_specs,requested_ids=None,missing_local='none',missing_optional='none'):
    if missing_optional not in ('none','partial','all'):raise ValueError('Invalid optional diagnostic')
    result,telemetry=infer_f0(pack_path,{k:cache_specs[k] for k in ('det','miw')},requested_ids,missing_local)
    pack,members,_=load(pack_path);names=result['clip_ids'];parent=result['fused'].copy();result['F0']=parent
    if not pack['optional']:return result,telemetry
    if set(pack['optional'])!={'CSN'}:raise ValueError('This inference entry requires CSN optional member')
    spec=pack['optional']['CSN']['spec'];cs=cache_specs.get('det248');present=np.zeros(len(names),dtype=bool);q=np.empty((0,40),dtype=np.float32);t=time.monotonic()
    if cs is not None and Path(cs['path']).with_name(Path(cs['path']).stem+'_meta.npz').exists():
        data,off,ids,build=load_fused(Path(cs['path']));assert isinstance(data,np.memmap) and data.shape[-3:]==(248,248,4)
        clashes,_=import_module('26_build_test_caches').compare_builds(cs['build'],list(build));assert not clashes,clashes
        lookup={str(c):i for i,c in enumerate(ids)};order=np.asarray([lookup.get(str(c),-1) for c in names]);present=np.asarray([i>=0 and off[i+1]>off[i] for i in order])
        if missing_optional=='partial':present[np.arange(len(names))%7==0]=False
        elif missing_optional=='all':present[:]=False
        if present.any():
            rows=order.copy();rows[~present]=order[present][0]
            ds=FusedFrameDataset(data,off,np.zeros(len(off)-1,dtype=np.int64),rows,32,False,make_fused_transform(224,erase_prob=0,flip_prob=0),42)
            net=model_from_spec(spec,members['CSN']).cuda().eval();q=predict(net,ds,'cuda',spec['batch_size'],True)[present];del net,ds;torch.cuda.empty_cache()
    fused=parent.copy();fused[present]=.8*parent[present]+.2*q.astype(np.float64)
    result.update(CSN=q,CSN_ids=names[present],optional_present=present,fused=fused)
    telemetry.update(optional_seconds=time.monotonic()-t,missing_optional=missing_optional,optional_present=int(present.sum()),optional_batch=spec['batch_size'],optional_channels=4,optional_crop=224,optional_fallback_exact=bool(np.array_equal(fused[~present],parent[~present])),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    return result,telemetry
