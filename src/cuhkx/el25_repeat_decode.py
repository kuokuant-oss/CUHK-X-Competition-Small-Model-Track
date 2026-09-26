"""Repeat-aware decoding context (decoder version repeat-v1).

A recording trial consists of consecutive takes that repeat the same action sequence. Takes on the same date whose
first clip starts less than ``gap`` seconds after the previous take's last clip start and that have the same number of
visible clips are chained. Inside a chain, each take's per-position log-probabilities are augmented with ``lam`` times the
sum of the other takes' log-probabilities at the same position (soft borrowing, not hard tying). The result is
renormalised per clip and decoded by the unchanged original MAP decoder (``decode_take`` with the packaged transition
prior). Single-take chains are untouched, so the original mode is reproduced exactly when no chain exists. When take
coverage is below the original decoder's .5 threshold the original raw fallback applies and nothing is borrowed.
No labels and no test outcomes are involved.
"""
# Role: repeat-chain borrowing. Consecutive takes of one date with equal clip counts, each starting
# less than GAP seconds after the start of the previous take's last clip, form a chain; each take's
# log-probabilities then get LAMBDA times the sum of the other chained takes' log-probabilities at
# the same position, and are renormalised per clip.
# Used by: scripts/el25r_p1_repeat_runtime.py, which applies repeat_context to the fused
# probabilities before h_mpm_runtime.submission_frame decodes them into prediction.csv; inference.
import numpy as np
# LAMBDA is the borrowing weight; the next take must start less than GAP seconds after the start of
# the previous take's last clip; VERSION is the name written to the report; MIN_COVERAGE is the
# share of clips that must lie in takes (the same 0.5 threshold as the decoder).
LAMBDA=.25;GAP=120.;VERSION='repeat-v1';MIN_COVERAGE=.5
def chains_from_table(table,gap=GAP):
    # table is the take table from el21_take_runtime.take_frame: one row per clip, where seconds is
    # the clip's start time. An empty table gives no chains.
    if table is None or len(table)==0:return []
    # One row per take, indexed by take id: its date, start = the start of its first clip, end =
    # the start of its last clip (the largest clip start), and n = its number of clips.
    tk=table.groupby('take_id').agg(date=('date','first'),start=('seconds','min'),end=('seconds','max'),n=('clip_id','size'))
    chains=[]
    # Each date is scanned separately, with its takes in start order.
    for _,g in tk.groupby('date'):
        # cur is the chain being built; it starts with the earliest take of the date.
        g=g.sort_values('start');cur=[g.index[0]]
        for prev,cur_id in zip(g.index[:-1],g.index[1:]):
            # prev is always the last take of cur. The next take joins cur if it starts less than
            # gap seconds after the start of prev's last clip and has as many clips as prev;
            # otherwise cur is closed and a new chain starts with this take.
            if (g.loc[cur_id,'start']-g.loc[prev,'end']<gap) and g.loc[cur_id,'n']==g.loc[prev,'n']:cur.append(cur_id)
            else:chains.append(cur);cur=[cur_id]
        chains.append(cur)
    # Only chains of two or more takes are returned, each as a list of take ids.
    return [list(map(str,c)) for c in chains if len(c)>1]
def repeat_context(clip_ids,probs,table,lam=LAMBDA,gap=GAP,min_coverage=MIN_COVERAGE):
    """Return (adjusted_probs, report). Rows of clips outside multi-take chains are returned unchanged; below the
    coverage threshold everything is returned unchanged so the original raw fallback is preserved."""
    # Step 1: index the clip ids (they must be unique), then check that probs is a finite
    # (n_clips, 40) array whose row i belongs to clip_ids[i]; p is its float64 version.
    names=list(map(str,clip_ids));ix={c:i for i,c in enumerate(names)};assert len(ix)==len(names)
    p=np.asarray(probs,dtype=np.float64);assert p.shape==(len(names),40) and np.isfinite(p).all()
    # Step 2: out starts as a copy of p. coverage is the share of clip_ids that appear in the take
    # table; chains are formed only if it reaches min_coverage, so below it nothing is borrowed.
    out=p.copy();covered=0 if table is None or len(table)==0 else sum(str(c) in ix for c in table.clip_id);coverage=covered/max(len(names),1)
    chains=chains_from_table(table,gap) if coverage>=min_coverage else [];touched=0
    if chains:
        # Step 3: members maps take id -> its clip ids in position order; logs maps take id -> its
        # (clips, 40) log-probabilities (p clipped at 1e-12), only for takes whose clips all have
        # probability rows.
        members={str(t):[str(c) for c in g.sort_values('position').clip_id] for t,g in table.groupby('take_id')}
        logs={t:np.log(np.clip(p[[ix[c] for c in m]],1e-12,None)) for t,m in members.items() if all(c in ix for c in m)}
        for chain in chains:
            takes=[t for t in chain if t in logs]
            if len(takes)<2:continue
            # Chained takes have equal clip counts, so their positions line up one to one.
            assert len({logs[t].shape for t in takes})==1,'chained takes must have identical size'
            for t in takes:
                # Step 4, for each take t of the chain: v = log p_t + lam * (sum of log p_o over the
                # other takes o), position by position; then a softmax per clip (subtract the row
                # maximum, exponentiate, divide by the row sum). Thus q_t is proportional to
                # p_t * prod_o p_o ** lam. Every take reads the unmodified logs, so the order in
                # which the takes are processed does not matter.
                v=logs[t]+lam*sum(logs[o] for o in takes if o!=t);v=v-v.max(1,keepdims=True);q=np.exp(v);q/=q.sum(1,keepdims=True)
                # Step 5: the new rows replace the take's clips in out; touched counts them.
                for c,row in zip(members[t],q):out[ix[c]]=row;touched+=1
    # Every row, changed or not, must be a finite distribution that sums to one.
    assert np.isfinite(out).all() and np.max(abs(out.sum(1)-1))<1e-5
    # The report holds the parameters, the coverage, the chains (lists of take ids), the numbers of
    # chained takes and adjusted clips, and raw_argmax_changed, the number of clips whose argmax
    # changed.
    return out,dict(version=VERSION,lam=lam,gap_seconds=gap,same_size_required=True,coverage=coverage,min_coverage=min_coverage,below_coverage_unchanged=coverage<min_coverage,chains=chains,takes_in_chains=sum(len(c) for c in chains),clips_adjusted=touched,raw_argmax_changed=int((out.argmax(1)!=p.argmax(1)).sum()))
