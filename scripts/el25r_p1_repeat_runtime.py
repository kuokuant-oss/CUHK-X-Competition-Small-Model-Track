"""Standalone raw runtime for P1 with decoder repeat-v1 (EL25R). All learned tensors restored from one unchanged FD18 checkpoint; original MAP and repeat answers decoded from the same probabilities."""
# Role: inference entry point. Checks and loads weights/model.pt, rebuilds the person windows and
#   the three test caches from the raw clip folders, runs members B and C and the transformer
#   readout, fuses them, decodes each take and writes the submission CSV with a run record.
# Used by: inference.sh (called by run_inference.sh); inference.
# Main steps: 1) guards and checkpoint loading 2) depth palette table 3) cache rebuild
#   4) cache checks 5) availability 6) takes 7) members and readout 8) fusion and chain-free
#   decoding per masking case 9) repeat chains and final decoding 10) outputs.
import argparse,base64,hashlib,io,json,os,socket,subprocess,sys,time
from pathlib import Path
from importlib import import_module
# Package root; src/ (the cuhkx library) and scripts/ (imported with import_module, since their
# names start with digits) go first on the import path.
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
# One OpenMP/MKL/OpenBLAS thread for repeatable results; set before numpy and torch are first
# imported (inside the functions below).
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'

# Write obj as indented JSON, creating the parent folder.
def save(path,obj):Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(obj,indent=2),encoding='utf-8')
# SHA256 hex digest of a file.
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

# Guards (installed in step 1): torch.load may read only the checkpoint file, and network
# connections raise. Returns the log of checkpoint reads, which keeps growing as later loads pass.
def guards(pack):
    import torch
    # The real loader, the read log, and the one file that may be loaded.
    original=torch.load;reads=[];allowed=Path(pack).resolve()
    # Guarded torch.load: weights_only=True is required (tensors and plain containers only);
    # in-memory streams unpacked from the checkpoint pass; any other file fails the assert.
    def checked(source,*args,**kwargs):
        assert kwargs.get('weights_only') is True
        if not isinstance(source,io.BytesIO):
            # Resolve a path or an open file to its path, reject it unless it is the checkpoint,
            # and log the read.
            path=Path(source if isinstance(source,(str,Path)) else source.name).resolve();assert path==allowed,'External weights rejected';reads.append(str(path))
        return original(source,*args,**kwargs)
    # Replacement for socket connections: always raises.
    def denied(*args,**kwargs):raise RuntimeError('FD18 inference network is disabled')
    # Install both guards.
    torch.load=checked;socket.socket.connect=denied;socket.create_connection=denied
    # Self-test: loading another file name next to the checkpoint must fail with the guard's own
    # message (the file is never opened).
    try:torch.load(allowed.parent/'FD18_EXTERNAL_WEIGHT_DENIAL_PROBE.pt',weights_only=True)
    except AssertionError as error:assert str(error)=='External weights rejected'
    else:raise AssertionError('External weight guard did not reject the probe')
    # Self-test: a connection to localhost must fail with the network guard's message.
    try:socket.create_connection(('127.0.0.1',1))
    except RuntimeError as error:assert str(error)=='FD18 inference network is disabled'
    else:raise AssertionError('Network guard did not reject the probe')
    return reads

# Availability (step 5): which modalities decode for each test clip, judged by the cache loader
# load_clip_fused. D: it returns at least one frame (Depth_Color readable, IR files present);
# I: D holds and the IR frames were readable (not 'ir_dead'). The readout is blended into F0
# only where both hold (401 of the 405 test clips).
def availability():
    from concurrent.futures import ThreadPoolExecutor
    from cuhkx.paths import p
    from cuhkx.fused import load_clip_fused
    from cuhkx import depth as depth_codec
    # Palette table written in step 2.
    lut=depth_codec.load_lut(p('processed_root')/'depth_lut.npz')
    def one(clip):
        # Decode the clip as the cache builder does (same 32-frame cap) but produce only a 1x1
        # whole-frame output: just the decode statistics (e.g. 'ir_dead') are used.
        frames,stats=load_clip_fused(clip/'Depth_Color',clip/'IR',lut,size=(1,1),max_frames=32,whole_frame=True)
        return clip.name,dict(D=bool(len(frames)),I=bool(len(frames) and not stats.get('ir_dead',False)),decode=stats)
    # All SM_test_* folders in name order, on 8 threads: {clip name: {'D', 'I', 'decode'}}.
    return dict(ThreadPoolExecutor(max_workers=8).map(one,sorted(c for c in p('test_root').glob('SM_test_*') if c.is_dir())))

def main():
    import numpy as np,pandas as pd,torch
    # Library parts: p() reads configs/paths.yaml; load and unpack open the checkpoint; infer runs
    # members B and C and the transformer readout; export_memmap writes a memory-mappable copy of
    # a cache; restore_state and check_state decompress a stored state dict and compare it with
    # its SHA256 manifest; submission_frame decodes each take into the CSV table; from_raw builds
    # the take table from the filename clocks.
    from cuhkx.paths import p
    from cuhkx.fd18_codec import load,unpack
    from cuhkx.el22_support_inference import infer
    from cuhkx.fused import export_memmap
    from cuhkx.fd_lzma_codec import restore_state
    from cuhkx.fd_codec import check_state
    from cuhkx.h_mpm_runtime import submission_frame
    from cuhkx.el22_clock_adapter import from_raw
    # Options: --pack is the checkpoint; --output-dir the run folder; --rebuild builds the person
    # windows and caches from the raw folders; --detector-only is the internal person-window step
    # of a rebuild; --active-readout names the readout to deliver (the checkpoint stores only
    # S1-G, so the default N0-G fails the check in step 7; inference.sh passes S1-G);
    # --all-readouts evaluates every readout stored in the checkpoint.
    ap=argparse.ArgumentParser();ap.add_argument('--pack',required=True);ap.add_argument('--output-dir',required=True);ap.add_argument('--rebuild',action='store_true');ap.add_argument('--detector-only',action='store_true');ap.add_argument('--active-readout',default='N0-G');ap.add_argument('--all-readouts',action='store_true')
    # --missing-local, --missing-i and --missing-d are accepted but not read by this script.
    for k in ['local','i','d']:ap.add_argument('--missing-'+k,choices=['none','partial','all'],default='none')
    # Step 1 (loading): parse the options; resolve the checkpoint and run-folder paths; create the
    # run folder and refuse one that already holds a finished run (complete.json); install the
    # guards; start the wall clock; load the checkpoint, which checks every SHA256 manifest and
    # rebuilds member C from B and the stored difference (blob: the stored parts, eq: int8 ViT
    # encoder, heads: readout heads, bm: members B and C, det: detector); meta: the members'
    # metadata (cache recipes, palette table, transition prior); require a full-data model
    # (fold -1) and a file below 100,000,000 bytes.
    a=ap.parse_args();pack=Path(a.pack).resolve();out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=True);assert not (out/'complete.json').exists();reads=guards(pack);start=time.perf_counter();blob,eq,heads,bm,det=load(pack);meta=blob['base']['meta'];assert blob['meta']['fold']==-1 and pack.stat().st_size<100000000
    # Step 2: decode the depth palette table stored in the metadata, check its SHA256 and write it
    # to processed_root/depth_lut.npz, or require an existing copy to be byte-identical; the
    # person-window step, the cache builder and availability() read it from there.
    lut=base64.b64decode(meta['depth_lut_base64']);assert hashlib.sha256(lut).hexdigest()==meta['depth_lut_sha256'];lutpath=p('processed_root')/'depth_lut.npz';lutpath.parent.mkdir(parents=True,exist_ok=True)
    if lutpath.exists():assert lutpath.read_bytes()==lut
    else:lutpath.write_bytes(lut)
    # Detector-only mode (the first command of step 3, run as a separate process): compute the
    # person windows with the checkpoint's detector, then stop.
    if a.detector_only:
        original=torch.load
        # torch.load replacement for the person-window script: read the checkpoint through the
        # guard, unpack the zlib envelope, decompress the detector state, check it against its
        # SHA256 manifest and return it in the {'components': {'detector': ...}} layout that
        # cuhkx.person.load_detector accepts.
        def detector_adapter(*args,**kwargs):
            value=unpack(original(*args,**kwargs),loader=original);base=value['base'];state=restore_state(base['detector']);check_state(state,base['tensor_manifest']['detector']);return dict(components={'detector':state})
        # Install it, run scripts/22_person_windows.py in this process on the test split (it
        # writes processed_root/person_windows_test.parquet) and log the checkpoint reads.
        torch.load=detector_adapter;sys.argv=['22_person_windows.py','--split','test','--detector-weights',str(pack)];import_module('22_person_windows').main();save(out/'detector-reads.json',dict(weight_reads=reads,network_disabled=True));return
    # Cache recipes from the checkpoint: det and miw from member C's metadata, det248 from the
    # top-level metadata; each gives the cache file name ('test_cache') and its build arguments
    # ('test_cache_build'). builder provides the recipe helpers of scripts/26_build_test_caches.py;
    # commands logs every subprocess command line.
    views=dict(meta['members']['C']['views']);views['det248']=blob['meta']['det248_cache'];builder=import_module('26_build_test_caches');commands=[]
    # Step 3 (cache rebuild, with --rebuild): the person windows, then the three caches.
    if a.rebuild:
        # The cache folder must be empty, so that no stale cache can be used.
        p('cache_root').mkdir(parents=True,exist_ok=True);assert not list(p('cache_root').iterdir())
        # Person windows: this script again with --detector-only, in a fresh interpreter.
        command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--pack',str(pack),'--output-dir',str(out),'--detector-only'];commands.append(command);subprocess.run(command,cwd=ROOT,check=True)
        for v in ['det','miw','det248']:
            # Recipe to command line: split_cache_name maps e.g. fused-det248_test.npz to builder
            # 'fused' and suffix '-det248'; argv_from_build turns the recorded arguments into
            # options (det and miw at 144 px, det248 at 248 px, miw adds --motion-in-window);
            # command[2], the builder script, becomes fd18_raw_builder.py (the same builder with
            # the network disabled and an availability record); add -B; log and run it.
            spec=views[v];prefix,suffix=builder.split_cache_name(spec['test_cache']);command=builder.argv_from_build(spec['test_cache_build'],prefix,suffix);command[2]=str(ROOT/'scripts/fd18_raw_builder.py');command.insert(1,'-B');commands.append(command);subprocess.run(command,cwd=ROOT,check=True)
    # Step 4: check each cache and collect its path and recipe for the recognition step.
    specs={}
    for v in ['det','miw','det248']:
        # Cache file named by the recipe, e.g. cache_root/fused-det_test.npz.
        spec=views[v];cache=p('cache_root')/spec['test_cache']
        # After a rebuild, write the memory-mappable copy (<name>_data.npy and the small
        # <name>_meta.npz) that the recognition step reads; otherwise it must already exist.
        if a.rebuild:export_memmap(cache)
        else:assert cache.with_name(cache.stem+'_data.npy').exists()
        # The build record saved with the cache must contain every recipe entry with the same
        # value (the per-source tally inside the 'windows=' entry is ignored).
        built=builder.recorded_build(cache);assert built is not None;clashes,_=builder.compare_builds(spec['test_cache_build'],built);assert not clashes,clashes
        specs[v]=dict(path=str(cache),build=spec['test_cache_build'])
    # Step 5 (availability): D/I flags from a fresh decode of the raw folders; the runtime expects
    # exactly the 405 test clips. Saved as raw-availability.json.
    av=availability();assert len(av)==405;save(out/'raw-availability.json',dict(source='fresh original load_clip_fused decode diagnostics, same cap32, no cache-value heuristic',availability=av))
    # Step 6 (takes): group the clips into takes by recorder start time (first timestamp - 100 ms
    # x frame counter, from the frame file names; a clip without an exact 100 ms clock stays out
    # of all takes), ordered by (start, end); save the grouping report and the take table
    # (clip_id, take_id, position, take_size, date, seconds, frame).
    table=from_raw(p('test_root'));save(out/'take-clock-report.json',table.attrs['clock_grouping']);table.to_parquet(p('processed_root')/'takes_test.parquet',index=False)
    # Step 7 (members and readout).
    from cuhkx.fd18_readout import compose,MASKS
    # Readouts to evaluate: every stored one with --all-readouts, else only --active-readout; each
    # must be stored in the checkpoint.
    candidates=blob['meta']['selected'] if a.all_readouts else [a.active_readout]
    assert all(c in blob['meta']['selected'] for c in candidates)
    # Members B (det) and C (det and miw) with four-pass averaging give F0; the frozen ViT-S
    # encodes det248 (IR and Depth_Color separately, plain and flipped), its tokens are pooled over
    # five regions with in-frame support weights, and the readout head gives q. results: the
    # fused output per readout (not used below); base: member probabilities, F0 and presence
    # flags; readouts: q per readout family; features: pooled features; telemetry: timings and
    # peak GPU memory.
    results,base,readouts,features,telemetry=infer(pack,specs,av,candidates)
    # Save the member outputs, readout probabilities, pooled features and the take table.
    np.savez(out/'components.npz',**base);np.savez(out/'readouts.npz',**readouts);np.savez(out/'features.npz',**features);table.to_parquet(out/'takes_test.parquet',index=False)
    # Step 8 (fusion and decoding without repeat chains, per masking case): the sample submission
    # gives the row order ('utf-8-sig' drops a byte-order mark); evidence collects output hashes.
    template=pd.read_csv(p('sample_submission'),encoding='utf-8-sig');evidence={}
    for cid in candidates:
        evidence[cid]={}
        # MASKS: 'warm' masks nothing (the delivered case); the other eight mark IR ('I...'), depth
        # ('D...') or the miw view ('local...') as missing on every 7th clip ('...partial') or on
        # all clips ('...all'; 'newall' drops IR and depth, 'allmissing' all three), to exercise
        # the fallbacks: a clip without IR or depth keeps F0; without miw, C uses its det output.
        for phase in MASKS:
            # Fuse the readout into F0 where IR and depth are present, for S1-G as
            # f = normalize(exp(0.8 log F0 + 0.2 log q)) with probabilities clipped at 1e-12, else
            # f = F0; save the result in <run folder>/<readout>/<phase>/probabilities.npz.
            result=compose(base,readouts,cid,phase);d=out/cid/phase;d.mkdir(parents=True,exist_ok=False);np.savez(d/'probabilities.npz',**result)
            # Decode each take jointly (distinct labels, transition prior at weight 0.75) without
            # repeat chains; clips outside takes (all clips, if fewer than half are in takes) keep
            # their argmax. Check for 405 distinct rows; write prediction.csv with Unix line ends.
            csv,report=submission_frame(result['clip_ids'],result['fused'],table,np.asarray(meta['prior']),template,mode='original');assert len(csv)==405 and csv.path.nunique()==405;csv.to_csv(d/'prediction.csv',index=False,lineterminator='\n')
            # SHA256 of the CSV, of the prediction column as int64 bytes and of the probabilities.
            evidence[cid][phase]=dict(csv_sha256=sha(d/'prediction.csv'),vector_sha256=hashlib.sha256(csv.prediction.to_numpy(dtype=np.int64).tobytes()).hexdigest(),probability_sha256=sha(d/'probabilities.npz'))
    # Step 9 (repeat chains and the final decode).
    # The published entrypoint's chosen vector is an actual output of this run.
    from cuhkx.el25_repeat_decode import repeat_context,VERSION,LAMBDA,GAP
    # Delivered probabilities: the active readout with nothing masked; prior: the 40x40 transition
    # matrix from the checkpoint; original: the decode without repeat chains.
    final=compose(base,readouts,a.active_readout,'warm');prior=np.asarray(meta['prior'],dtype=np.float64);original,report0=submission_frame(final['clip_ids'],final['fused'],table,prior,template,mode='original')
    # Repeat chains: consecutive same-date takes with equal clip counts, each starting less than
    # GAP = 120 s after the start of the previous take's last clip. In a chain, each take's
    # log-probabilities gain LAMBDA = 0.25 times the sum of the other takes' log-probabilities at
    # the same position, renormalised per clip (nothing changes when fewer than half of the clips
    # are in takes). The adjusted probabilities are then decoded again.
    adjusted,context=repeat_context(final['clip_ids'],final['fused'],table,LAMBDA,GAP);csv,report=submission_frame(final['clip_ids'],adjusted,table,prior,template,mode='original')
    # The chain-free decode must equal the 'warm' CSV of step 8, and the final table must have 405
    # distinct rows with labels 0-39.
    assert original.equals(pd.read_csv(out/a.active_readout/'warm/prediction.csv')) and len(csv)==405 and csv.path.nunique()==405 and csv.prediction.between(0,39).all()
    # Step 10 (outputs): prediction-original.csv (no repeat chains) and prediction.csv (the
    # submission; run_inference.sh copies it to --output); hv: SHA256 of a table's prediction
    # column as int64 bytes, the vector hash that run_inference.sh prints.
    original.to_csv(out/'prediction-original.csv',index=False,lineterminator='\n');csv.to_csv(out/'prediction.csv',index=False,lineterminator='\n');hv=lambda fr:hashlib.sha256(fr.prediction.to_numpy(dtype=np.int64).tobytes()).hexdigest()
    # Summary: output hashes, rows changed by the repeat chains, and the decoder settings, chains
    # and reports.
    extra=dict(vector_sha256=hv(csv),original_vector_sha256=hv(original),changed_vs_original=int((csv.prediction!=original.prediction).sum()),csv_sha256=sha(out/'prediction.csv'),original_csv_sha256=sha(out/'prediction-original.csv'),decoder=dict(version=VERSION,lam=LAMBDA,gap_seconds=GAP,chains=context['chains'],clips_adjusted=context['clips_adjusted'],original_report=report0,report=report))
    # complete.json marks the run as finished (step 1 then refuses this run folder) and also logs
    # the checkpoint hash and size, wall time, checkpoint reads, subprocess commands, telemetry and
    # the step 8 hashes.
    save(out/'complete.json',dict(**extra,passed=True,model_sha256=sha(pack),model_bytes=pack.stat().st_size,wall_seconds=time.perf_counter()-start,weight_reads=reads,external_weight_reads=0,network_disabled=True,raw_rebuild=a.rebuild,telemetry=telemetry,commands=commands,candidates=candidates,active_readout=a.active_readout,evidence=evidence,components_mask_independent=True))
    # One-line JSON summary on stdout.
    print(json.dumps(dict(passed=True,candidates=candidates,seconds=time.perf_counter()-start,telemetry=telemetry)),flush=True)
if __name__=='__main__':main()
