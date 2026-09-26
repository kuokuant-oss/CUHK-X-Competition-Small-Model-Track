"""Frozen exploratory subject-fold scoring with all baselines and window controls."""
# Role: fold scoring. `measure` decodes each configuration's probabilities for the clips of every
#   validation fold per take (joint MAP or marginal decoding with that fold's transition prior)
#   and reports raw and decoded accuracy per population, complete takes included, with
#   per-length, per-clip and per-subject tables; `gate` applies a fixed acceptance rule to its
#   result. `z` and `align` load and reorder saved probabilities.
# Used by: el21_candidates.py, which passes `z` and `align` on to el22_p1_heads.py; training.
#   `measure` and `gate` are not called by any script in this package.
import json
from pathlib import Path

from el20_common import DEST, ROOT, archive_code, j, now, sha, sources, write
from iterative_common import populations


# Load an .npz file (pickled objects refused) into a dict of in-memory arrays.
def z(path):
    import numpy as np
    with np.load(path,allow_pickle=False) as f: return {k:f[k] for k in f.files}


# Rows of blob[key] (default 'probs') in the order of `ids`, as float64; the file's clip ids
# must be unique and include every requested id.
def align(blob,ids,key='probs'):
    import numpy as np
    names=list(map(str,blob['clip_ids'])); assert len(names)==len(set(names))
    ix={c:i for i,c in enumerate(names)}; assert set(ids)<=set(ix)
    return blob[key][[ix[c] for c in ids]].astype(np.float64)


# Inputs: stage = name of the output folder under DEST (must be new); predictions = {fold:
# (clip ids, {source name: (clips, 40) probabilities})}; configs = {configuration name: dict with
# the probability 'source', the 'decoder' ('MPM', else joint MAP) and optional
# 'prototype_source', 'parent' and 'additional_parents'}; source_paths = extra files whose SHA256
# is recorded; control_aliases = {configuration: its name in an earlier fold-results table}, for
# configurations whose scores must reproduce that table.
def measure(stage,predictions,configs,source_paths,control_aliases=None):
    import numpy as np
    import pandas as pd
    from cuhkx.takes import decode_take
    from cuhkx.h_mpm_prototype import mpm
    out=DEST/stage; out.mkdir(parents=True,exist_ok=False)
    # Record the SHA256 of this code and its dependencies; they are checked again at the end.
    source_paths=[Path(__file__),ROOT/'training/el20_common.py',ROOT/'training/iterative_common.py',ROOT/'training/next_common.py',ROOT/'src/cuhkx/takes.py',ROOT/'src/cuhkx/h_mpm_prototype.py',*source_paths]
    manifest=sources(source_paths); write(out/'sources.json',manifest)
    # Clip table, per-fold populations and priors, and length weights (next_common.populations);
    # `arms` = the configuration names, followed by the accumulators for the output tables.
    table,contexts,weights=populations(); arms=list(configs); allrows=[]; events=[]; levels=[]; controls=[]
    # The earlier fold-results table, needed only for the reproduction checks.
    old=pd.read_csv(DEST/'external-evidence/2026-09-12-final-capacity-probes-folds.csv') if control_aliases else None
    for fold,(ids,arrays) in predictions.items():
        # The ids must be exactly the fold's clips; every array must be (clips, 40), finite,
        # non-negative, with rows summing to 1 (within 1e-5).
        ids=list(map(str,ids)); assert len(ids)==len(set(ids)) and set(ids)==set(table[table.fold==fold].index)
        for q in arrays.values():
            assert q.shape==(len(ids),40) and np.isfinite(q).all() and (q>=0).all() and np.max(abs(q.sum(1)-1))<1e-5
        # Row of each clip id, this fold's populations and prior, and a cache of scored windows.
        ix={c:i for i,c in enumerate(ids)}; ctx=contexts[fold]; cache={}
        # Score one window (the ordered clip ids of a take or thinned take), memoised: for each
        # configuration the argmax labels and the labels decoded jointly over the window.
        # Returns (values, decoded, raw); values[i] = (raw, decoded accuracy) of configuration i.
        def evaluate(w):
            key=tuple(w)
            if key in cache:return cache[key]
            rows=[ix[c] for c in w];y=table.loc[list(w),'action_id'].to_numpy();decoded={};raw={}
            for a,c in configs.items():
                q=arrays[c['source']][rows]
                # 'MPM': per-clip mode of the distinct-sequence marginals (h_mpm_prototype.mpm);
                # otherwise the joint MAP beam search used at inference, transition weight 0.75.
                decoded[a]=mpm(q,ctx['prior']) if c['decoder']=='MPM' else decode_take(q,ctx['prior'],.75)
                raw[a]=q.argmax(1)
                # With a prototype source, that array must decode to the same labels in each window.
                proto=c.get('prototype_source')
                if proto:
                    p=arrays[proto][rows]; pred=decoded[a] if np.array_equal(p,q) else mpm(p,ctx['prior']) if c['decoder']=='MPM' else decode_take(p,ctx['prior'],.75)
                    assert np.array_equal(decoded[a],pred),f'Actual/prototype window differs fold{fold}/{a}/{w}'
            values=np.array([[(raw[a]==y).mean(),(decoded[a]==y).mean()] for a in arms])
            cache[key]=(values,decoded,raw);return cache[key]
        # Clip-weighted mean of the window values over one population ({take id: clip ids});
        # with keep=True one record per clip is kept for the per-clip and per-subject tables.
        def agg(takes,keep=False):
            sums=np.zeros((len(arms),2)); n=0
            for w in takes.values():
                vals,dec,raw=evaluate(w);sums+=vals*len(w);n+=len(w)
                if keep:
                    for i,c in enumerate(w):
                        events.append(dict(fold=fold,subject=str(table.loc[c,'user']),clip_id=c,label=int(table.loc[c,'action_id']),length=len(w),**{a:int(dec[a][i]) for a in arms},**{a+'_raw':int(raw[a][i]) for a in arms}))
            return sums/n
        # Populations: the complete takes, the test-like subset (per-clip records kept) and the
        # mean over the 20 contiguous test-sized thinnings.
        values={'full_take':agg(ctx['takes']),'primary':agg(ctx['primary'],True),'designated20':np.mean([agg(v) for v in ctx['designated']],axis=0)}
        # Every contiguous window of length 1-8 inside the complete takes: mean accuracy per take,
        # then the mean over the takes that are long enough.
        lengthvalues=[]
        for length in range(1,9):
            per_take=[np.mean([evaluate(m[i:i+length])[0] for i in range(len(m)-length+1)],axis=0) for m in ctx['takes'].values() if len(m)>=length]
            level=np.mean(per_take,axis=0);lengthvalues.append(level)
            for i,a in enumerate(arms):levels.append(dict(fold=fold,length=length,arm=a,raw=float(level[i,0]),decoded=float(level[i,1]),takes=len(per_take)))
        # 'exact_length' combines the eight per-length accuracies with the length weights.
        values['exact_length']=np.tensordot(weights,np.stack(lengthvalues),axes=(0,0))
        # One row per population and configuration; configurations with an alias must match the
        # earlier table to within 1e-12.
        for pop,vals in values.items():
            for i,a in enumerate(arms):
                if control_aliases and a in control_aliases:
                    ref=old[(old.fold==fold)&(old.population==pop)&(old.arm==control_aliases[a])].iloc[0]
                    assert abs(vals[i,0]-ref.raw)<1e-12 and abs(vals[i,1]-ref.decoded)<1e-12,(fold,pop,a,vals[i],ref)
                allrows.append(dict(fold=fold,population=pop,arm=a,raw=float(vals[i,0]),decoded=float(vals[i,1])))
        controls.append(dict(fold=fold,windows=len(cache),all_prototype_answers_exact=True,original_controls_exact=True))
        print(json.dumps(dict(stage=stage,fold=fold,windows=len(cache),complete=True)),flush=True)
    # Summary per configuration and population: means over the folds of raw and decoded accuracy
    # and, against each reference configuration that was scored too (BNcal, H-MPM, its `parent`
    # (default F0-MAP), F0-MAP and any `additional_parents`), the gain in percentage points (pp):
    # mean, per fold, number of folds with a positive decoded gain, and the worst fold.
    df=pd.DataFrame(allrows);ev=pd.DataFrame(events);lf=pd.DataFrame(levels)
    summary={};subjects=[]
    for a in arms:
        summary[a]={}
        for pop in ('full_take','primary','designated20','exact_length'):
            g=df[(df.arm==a)&(df.population==pop)].sort_values('fold')
            summary[a][pop]=dict(raw=float(g.raw.mean()),decoded=float(g.decoded.mean()),contrasts={})
            for base in dict.fromkeys(['BNcal','H-MPM',configs[a].get('parent','F0-MAP'),'F0-MAP',*configs[a].get('additional_parents',[])]):
                if base not in arms: continue
                b=df[(df.arm==base)&(df.population==pop)].sort_values('fold');dd=100*(g.decoded.to_numpy()-b.decoded.to_numpy());rr=100*(g.raw.to_numpy()-b.raw.to_numpy())
                summary[a][pop]['contrasts'][base]=dict(decoded_pp=float(dd.mean()),raw_pp=float(rr.mean()),fold_decoded_pp=dd.tolist(),fold_raw_pp=rr.tolist(),positive_folds=int((dd>0).sum()),worst_fold_pp=float(dd.min()))
        # Per subject on the test-like subset: raw and decoded correct counts, and per reference
        # (BNcal, H-MPM, F0-MAP, any `additional_parents`) the decoded clips this configuration
        # gets right and the reference gets wrong (rescued), or the reverse (lost).
        for (f,sub),g in ev.groupby(['fold','subject']):
            row=dict(arm=a,fold=int(f),subject=str(sub),n=len(g),raw_correct=int((g[a+'_raw']==g.label).sum()),decoded_correct=int((g[a]==g.label).sum()))
            for base in dict.fromkeys(['BNcal','H-MPM','F0-MAP',*configs[a].get('additional_parents',[])]):
                if base in arms:
                    ac,bc=g[a]==g.label,g[base]==g.label;row[base+'_rescued']=int((ac&~bc).sum());row[base+'_lost']=int((~ac&bc).sum())
            subjects.append(row)
    # Write the four tables, check that no recorded source changed during the run, and write
    # results.json.
    df.to_csv(out/'folds.csv',index=False);ev.to_csv(out/'primary.csv',index=False);lf.to_csv(out/'lengths.csv',index=False);pd.DataFrame(subjects).to_csv(out/'subjects.csv',index=False)
    for p,h in manifest.items():assert sha(p)==h
    result=dict(at=now(),stage=stage,scope='Repeatedly used exploratory OOF; not independent validation or LB estimate',folds=list(predictions),configurations=configs,summary=summary,controls=controls,source_sha256=manifest)
    write(out/'results.json',result);return result


# Acceptance rule for configuration `arm` against reference `parent`: all four folds scored; on
# the 20 thinnings a mean decoded gain >= threshold, at least `positive` folds with a gain and no
# fold below `worst`; no mean decoded loss on the test-like subset; and a length-weighted
# decoded gain >= `exact` (all in pp). `kind` selects (threshold, exact, positive, worst):
# 'root' (0.5, 0, 3, -1), 'increment' (0.2, 0, 3, -0.5), 'exploratory' (0.2, -0.1, 2, -1).
def gate(result,arm,parent,kind):
    v=result['summary'][arm]; d=v['designated20']['contrasts'][parent]; p=v['primary']['contrasts'][parent];e=v['exact_length']['contrasts'][parent]
    threshold,exact,positive,worst={'root':(.5,0,3,-1),'increment':(.2,0,3,-.5),'exploratory':(.2,-.1,2,-1)}[kind]
    return len(result['folds'])==4 and d['decoded_pp']>=threshold and p['decoded_pp']>=0 and e['decoded_pp']>=exact and d['positive_folds']>=positive and d['worst_fold_pp']>=worst

