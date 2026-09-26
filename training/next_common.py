"""N0912 fixed populations, independent parent controls and registered gates."""
# Role: builds the validation-fold populations (`populations`: per fold the complete takes, the
#   test-like subset, 20 test-sized thinnings and the transition prior) and holds a paired scorer
#   of a candidate against its parent configuration (`score`) with source-manifest helpers.
# Used by: iterative_common.populations, which runs only this file's `populations` function;
#   training. `freeze`, `verify` and `score` are not called by any script in this package.
import ast
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Single-threaded BLAS for reproducible numerics; effective only if numpy is imported afterwards.
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
# ROOT: the project root, one level above training/.
ROOT = Path(__file__).resolve().parents[1]
# Make cuhkx (src/), the numbered scripts and the training modules importable.
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'),str(ROOT/'training')]
# DEST: output folder of one development iteration; REVIEW: review root from $CUHKX_REVIEW_ROOT
# (default ./external-review).
DEST = ROOT / 'models/next-20260912'
REVIEW = Path(os.environ.get('CUHKX_REVIEW_ROOT', 'external-review'))


# Read a JSON file.
def j(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


# Plain JSON write (not atomic).
def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


# SHA256 hex digest of a file.
def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


# Load an .npz file (pickled objects refused) into a dict of in-memory arrays.
def z(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


# `arm` names the configuration being evaluated. Check that the files listed in
# DEST/<arm>/protocol.json are unchanged, then record the SHA256 of protocol.json, this file and
# `sources` in DEST/<arm>/<phase>-sources.json (mode 'x' never overwrites an existing manifest).
# Returns the manifest.
def freeze(arm, phase, sources):
    protocol = DEST / arm / 'protocol.json'
    for path, digest in j(protocol)['plan_and_source_sha256'].items():
        assert sha(path) == digest, path
    paths = list(dict.fromkeys([protocol, Path(__file__), *map(Path, sources)]))
    manifest = dict(created=datetime.now().astimezone().isoformat(), sources={str(p): sha(p) for p in paths})
    with (DEST / arm / (phase + '-sources.json')).open('x', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    return manifest


# Assert that every file in a manifest written by `freeze` still has its recorded SHA256.
def verify(manifest):
    for path, digest in manifest['sources'].items():
        assert sha(path) == digest, path


# Returns (table, folds, weights): the clip table, {fold: populations and prior} and the length
# weights. The validation protocol is described in METHOD.md section 7.
def populations():
    import numpy as np
    import pandas as pd
    from cuhkx.takes import transition_matrix
    # Step 1: clip table. folds.parquet (fold, user and action_id per clip) in the clip order of
    # oof.npz of the fold models behind the initialisation (ig65m-t32-det-60ep); there must be
    # 2,931 clips whose labels agree with that file.
    ref = z(ROOT / 'models/ig65m-t32-det-60ep/oof.npz')
    table = pd.read_parquet(ROOT / 'data/processed/folds.parquet').set_index('clip_id').loc[ref['clip_ids']]
    assert len(table) == 2931 and np.array_equal(table.action_id, ref['labels'])
    # Step 2: complete training takes, take id -> clip ids in recording order; 791 takes.
    train = pd.read_parquet(ROOT / 'data/processed/takes_train.parquet')
    takes = {str(k): g.sort_values('position').clip_id.tolist() for k, g in train[train.clip_id.isin(table.index)].groupby('take_id')}
    assert len(takes) == 791
    # Fold of each take (from its first clip); all clips of a take must lie in the same fold.
    tf = {k: int(table.loc[m[0], 'fold']) for k, m in takes.items()}
    assert all(set(table.loc[m, 'fold']) == {tf[k]} for k, m in takes.items())
    # Clip count of every test take; the thinned takes draw their lengths from these counts.
    target = pd.read_parquet(ROOT / 'data/processed/takes_test.parquet').groupby('take_id').size().to_numpy()
    # Step 3: take `subsample_contiguous` from training/95_fusion_shape_audit.py without importing
    # that script (which would run its module-level code): parse the file, compile only that
    # function and execute it in a namespace holding only `np`. It keeps one contiguous run of
    # each take, of a length drawn from `target` and capped at the take's length.
    module = ast.parse((ROOT / 'training/95_fusion_shape_audit.py').read_text(encoding='utf-8'))
    nodes = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'subsample_contiguous']
    assert len(nodes) == 1
    scope = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'original_subsample', 'exec'), scope)
    # One generator with seed 0 for all folds, so every call draws the same 20 thinnings.
    rng = np.random.default_rng(0)
    folds = {}
    # Step 4, per fold k: `takes` = the complete takes of fold k; `primary` = the test-like
    # subset, i.e. the thinned takes whose clips were pseudo-labelled for fold k's member-B
    # self-training (read from that run's identity.json, written by 101_matched_replay.py);
    # `designated` = 20 contiguous thinnings of fold k's takes; `prior` = the 40x40 transition
    # matrix counted on the other three folds' takes, with add-one smoothing.
    for fold in range(4):
        mine = {k: m for k, m in takes.items() if tf[k] == fold}
        folds[fold] = dict(takes=mine,
            primary=j(ROOT / f'models/abc-b-20260911/fold{fold}/identity.json')['registered']['pseudo_takes'],
            designated=[scope['subsample_contiguous'](mine, target, rng) for _ in range(20)],
            prior=transition_matrix({k: m for k, m in takes.items() if tf[k] != fold}, table.action_id.to_dict()))
    # One weight per window length 1-8, used to combine per-length accuracies into the
    # `exact_length` score; read from a JSON file under REVIEW.
    weights = np.array(j(REVIEW / '2026-09-12-H-decoder-precision.json')['exact_length_weights'])
    return table, folds, weights


def score(arm, stage, predictions):
    """predictions[fold] = (ids, same-extraction parent, candidate, archived parent)."""
    import numpy as np
    import pandas as pd
    from cuhkx.h_mpm_prototype import mpm
    from cuhkx.takes import decode_take
    # Paired comparison of a candidate configuration (named by `arm`) with its parent. Stage
    # 'pilot' scores folds 0 and 3, any other stage all four folds. Configuration N1 is decoded
    # with joint MAP decoding (weight 0.75); all others with `mpm`, the per-clip mode of the
    # distinct-sequence marginals (cuhkx/h_mpm_prototype.py).
    expected = [0, 3] if stage == 'pilot' else [0, 1, 2, 3]
    assert sorted(predictions) == expected
    table, context, weights = populations()
    # Six values per window: raw (argmax) and decoded accuracy of parent and candidate, and the
    # shares of clips the candidate gets right and the parent wrong after decoding (rescued),
    # or the reverse (lost).
    columns = ['parent_raw', 'candidate_raw', 'parent_decoded', 'candidate_decoded', 'rescued', 'lost']
    totals, events, controls = [], [], []
    decode = (lambda q, p: decode_take(q, p, .75)) if arm == 'N1' else mpm
    for fold in expected:
        ids, parent, candidate, archived = predictions[fold]
        # The ids must be exactly the fold's clips; every probability array must be (clips, 40),
        # finite, non-negative, with rows summing to 1 (within 1e-5).
        assert len(set(ids)) == len(ids) and set(ids) == set(table[table.fold == fold].index)
        for q in (parent, candidate, archived):
            assert q.shape == (len(ids), 40) and np.isfinite(q).all() and (q >= 0).all()
            assert np.max(np.abs(q.sum(1) - 1)) < 1e-5
        idx = {c: i for i, c in enumerate(ids)}
        ctx = context[fold]
        cache = {}
        # Score one window (the ordered clip ids of a take or thinned take), memoised; the parent
        # must decode exactly like its archived copy on every clip. Returns (values, records).
        def evaluate(window):
            key = tuple(window)
            if key in cache:
                return cache[key]
            ix = [idx[c] for c in window]
            truth = table.loc[window, 'action_id'].to_numpy()
            a, b = decode(parent[ix], ctx['prior']), decode(candidate[ix], ctx['prior'])
            ctrl = decode(archived[ix], ctx['prior'])
            # Strict per-clip control, including every designated window.
            assert np.array_equal(a, ctrl), f'{arm} fold{fold} archived parent mismatch {key}'
            ra, rb = parent[ix].argmax(1), candidate[ix].argmax(1)
            ac, bc = a == truth, b == truth
            vals = np.array([(ra == truth).mean(), (rb == truth).mean(), ac.mean(), bc.mean(), (bc & ~ac).mean(), (ac & ~bc).mean()])
            ev = [dict(fold=fold, subject=str(table.loc[c, 'user']), clip_id=c, label=int(truth[i]), parent_raw=int(ra[i]), candidate_raw=int(rb[i]), parent_decoded=int(a[i]), candidate_decoded=int(b[i])) for i, c in enumerate(window)]
            cache[key] = vals, ev
            return vals, ev
        # Clip-weighted mean of the window values over one population ({take id: clip ids});
        # with primary=True the per-clip records are kept as well.
        def aggregate(takes, primary=False):
            n, total = 0, np.zeros(len(columns))
            for w in takes.values():
                vals, ev = evaluate(w)
                n += len(w)
                total += len(w) * vals
                if primary:
                    events.extend(ev)
            return total / n
        # Populations: the test-like subset (per-clip records kept) and the mean over the 20
        # contiguous thinnings.
        for pop, values in [('primary', aggregate(ctx['primary'], True)), ('designated20', np.mean([aggregate(t) for t in ctx['designated']], axis=0))]:
            totals.append(dict(fold=fold, population=pop, **dict(zip(columns, values.tolist()))))
        # Four-fold stage only: every contiguous window of length 1-8 inside the complete takes,
        # averaged per take and then over the takes, combined with the length weights.
        if stage == 'fourfold':
            bylength = []
            for length in range(1, 9):
                per_take = [np.mean([evaluate(m[s:s+length])[0] for s in range(len(m)-length+1)], axis=0) for m in ctx['takes'].values() if len(m) >= length]
                assert per_take
                bylength.append(np.mean(per_take, axis=0))
            totals.append(dict(fold=fold, population='exact_length', **dict(zip(columns, (np.stack(bylength).T @ weights).tolist()))))
        # Per-fold checks: distinct windows scored, largest probability difference between the
        # parent and its archived copy, and the number of clips whose argmax differs between them.
        controls.append(dict(fold=fold, windows=len(cache), per_clip_parent_exact=True, max_probability_delta=float(np.max(np.abs(parent-archived))), raw_control_D=int((parent.argmax(1)!=archived.argmax(1)).sum())))
        print(json.dumps({'fold':fold,'rows':[r for r in totals if r['fold']==fold]}), flush=True)
    # Candidate minus parent in percentage points (pp) per fold, and the acceptance checks per
    # population: mean decoded gain >= 0.25 pp (pilot) or 0.5 pp, a positive decoded gain in at
    # least 1 (pilot) or 3 folds, no fold below -1 pp, and a mean raw gain >= -0.25 pp.
    deltas, gate = {}, {}
    for pop in ('primary', 'designated20'):
        rows = [r for r in totals if r['population'] == pop]
        decoded = [100*(r['candidate_decoded']-r['parent_decoded']) for r in rows]
        raw = [100*(r['candidate_raw']-r['parent_raw']) for r in rows]
        deltas[pop] = dict(decoded=decoded, raw=raw, mean_decoded=float(np.mean(decoded)), mean_raw=float(np.mean(raw)))
        gate[pop] = dict(decoded_mean=np.mean(decoded) >= (.25 if stage=='pilot' else .5), positive=sum(d>0 for d in decoded)>=(1 if stage=='pilot' else 3), worst=min(decoded)>=-1, raw_mean=np.mean(raw)>=-.25)
    # Four-fold stage: the mean length-weighted decoded gain must also be positive.
    if stage == 'fourfold':
        dd=[100*(r['candidate_decoded']-r['parent_decoded']) for r in totals if r['population']=='exact_length']
        deltas['exact_length']=dict(decoded=dd, mean_decoded=float(np.mean(dd)))
        gate['exact_length']={'mean_positive':np.mean(dd)>0}
    # Per-clip records of the test-like subset, each clip once; 1,631 clips in the four-fold stage.
    frame=pd.DataFrame(events)
    assert frame.clip_id.nunique()==len(frame)
    if stage=='fourfold': assert len(frame)==1631
    # Per subject: clips rescued and lost, net change, raw and decoded accuracies of both.
    subjects=[]
    for (fold, subject), g in frame.groupby(['fold','subject']):
        a,b=g.parent_decoded==g.label,g.candidate_decoded==g.label
        subjects.append(dict(fold=int(fold),subject=subject,n=len(g),rescued=int((b&~a).sum()),lost=int((a&~b).sum()),net=int(b.sum()-a.sum()),parent_raw=float((g.parent_raw==g.label).mean()),candidate_raw=float((g.candidate_raw==g.label).mean()),parent_decoded=float(a.mean()),candidate_decoded=float(b.mean())))
    # Descriptive bootstrap: 100,000 resamples of the subjects within each fold (seed 20260912);
    # each resample gives the net change per clip, which is then averaged over the folds.
    rng=np.random.default_rng(20260912)
    boot=[]
    for fold in expected:
        rows=[r for r in subjects if r['fold']==fold]
        ns=np.array([r['n'] for r in rows]);net=np.array([r['net'] for r in rows])
        ix=rng.integers(0,len(rows),(100000,len(rows)))
        boot.append(net[ix].sum(1)/ns[ix].sum(1))
    # Result: `eligible` only if every check passes; the bootstrap is summarised by its 2.5% and
    # 97.5% quantiles in pp.
    gate={p:{k:bool(v) for k,v in checks.items()} for p,checks in gate.items()}
    result=dict(arm=arm,stage=stage,eligible=all(all(c.values()) for c in gate.values()),gate=gate,deltas_pp=deltas,folds=totals,controls=controls,subjects=subjects,primary_rescued=sum(r['rescued'] for r in subjects),primary_lost=sum(r['lost'] for r in subjects),bootstrap_descriptive95_pp=np.quantile(100*np.mean(boot,axis=0),[.025,.975]).tolist(),bootstrap_note='100000 seed20260912 within-fold subject draws; reused OOF descriptive only, not LB',submission_ready=False)
    # Write the per-clip CSV and the result JSON, record the outcome in the state file under
    # reports/, append a summary to the progress log, and print it.
    frame.to_csv(DEST/arm/(stage+'-primary-per-clip.csv'),index=False)
    write(DEST/arm/(stage+'-result.json'),result)
    statepath=ROOT/'reports/2026-09-12-next-state.json';state=j(statepath)
    state[arm].update(status=('extension_pending' if stage=='pilot' else 'delivery_pending') if result['eligible'] else 'gate_failed_closed',result=str(DEST/arm/(stage+'-result.json')))
    write(statepath,state)
    with (ROOT/'reports/2026-09-12-next-progress.md').open('a',encoding='utf-8') as f:
        f.write('\n\n'+json.dumps({k:result[k] for k in ['arm','stage','eligible','deltas_pp','primary_rescued','primary_lost']},ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('subjects','folds')}),flush=True)
    return result
