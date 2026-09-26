"""Fixed experimental H-MPM decoder; not installed or submitted.

Uses the original transition prior and weight .75. Exact distinct-sequence
marginals for up to 3 clips; beam256/top40 for longer takes; pointwise argmax.
The caller retains its existing missing-take/probability fallback and validates
finite probabilities and class order. Output labels may repeat within a take.
No torch/model loading, filesystem or network operations occur on import.
"""
# Role: an alternative take decoder (maximum posterior marginal): each clip gets the argmax of its
# label marginal over sequences of pairwise distinct labels, scored with the transition prior at
# weight 0.75 (exact for takes of up to 3 clips, from a 256-beam search for longer takes).
# Used by: h_mpm_runtime.decode_predictions with mode='mpm', and the fold-scoring scripts in
# training/ (next_common.py, el20_scoring.py, fd17_scoring.py); not used by the delivered run,
# which decodes with mode='original' (takes.decode_take). h_mpm_runtime imports it, but none of its
# functions runs there.
from __future__ import annotations
import numpy as np
N_CLASSES=40
idx40=np.arange(40)
# valid3[a, b, c] is True when the labels a, b and c are pairwise distinct (shape 40x40x40).
valid3=(idx40[:,None,None]!=idx40[None,:,None]) & (idx40[:,None,None]!=idx40[None,None,:]) & (idx40[None,:,None]!=idx40[None,None,:])

def beams_for_take(probs, transitions, w, beam_width=64, candidates_per_step=16):
    """Exactly decode_take's search, but return the whole final beam instead of beams[0]."""
    n = len(probs)
    log_probs = np.log(np.clip(probs, 1e-12, None))
    # A single clip has one beam: its argmax, scored by its log-probability.
    if n == 1:
        return [(float(log_probs[0].max()), [int(log_probs[0].argmax())])]
    log_trans = np.log(np.clip(transitions, 1e-12, None))
    # The candidates of each clip are its candidates_per_step most probable classes.
    shortlist = [
        np.argsort(-log_probs[i])[: min(candidates_per_step, N_CLASSES)] for i in range(n)
    ]
    # Each beam is (score, labels so far, bitmask of used labels).
    beams = [(0.0, [], 0)]
    for i in range(n):
        # Extend each beam by each unused candidate of clip i; the score adds its log-probability
        # and, after the first clip, w times the log transition from the previous label.
        expanded = []
        for score, chosen, used in beams:
            for action in shortlist[i]:
                bit = 1 << int(action)
                if used & bit:
                    continue
                step = log_probs[i, action]
                if chosen:
                    step += w * log_trans[chosen[-1], action]
                expanded.append((score + step, [*chosen, int(action)], used | bit))
        # If every beam has already used all candidates of clip i, all 40 classes are tried.
        if not expanded:
            for score, chosen, used in beams:
                for action in range(N_CLASSES):
                    bit = 1 << action
                    if used & bit:
                        continue
                    step = log_probs[i, action]
                    if chosen:
                        step += w * log_trans[chosen[-1], action]
                    expanded.append((score + step, [*chosen, action], used | bit))
        # Keep the beam_width best partial sequences.
        expanded.sort(key=lambda b: -b[0])
        beams = expanded[:beam_width]
    # The result lists (score, labels) for every surviving beam, best first.
    return [(s, seq) for s, seq, _ in beams]

def marginals_from(beams, n):
    """Softmax the beam scores and accumulate per-position class mass."""
    # Scores are shifted by their maximum before exp for numerical stability.
    scores = np.array([s for s, _ in beams], dtype=np.float64)
    scores -= scores.max()
    wts = np.exp(scores)
    wts /= wts.sum()
    # m[i, c] is the total weight of the beams that give clip i the label c.
    m = np.zeros((n, N_CLASSES), dtype=np.float64)
    for wt, (_, seq) in zip(wts, beams, strict=True):
        for i, c in enumerate(seq):
            m[i, c] += wt
    return m

def full_short(p,prior):
    # Exact marginals for takes of 1-3 clips, by enumerating every label sequence; returns the
    # (n, 40) marginals and the best sequence.
    n=len(p)
    # e: (n, 40) log-probabilities; t: the 40x40 log transition prior times 0.75.
    e=np.log(np.clip(p,1e-12,None)).astype(np.float64)
    t=.75*np.log(np.clip(prior,1e-12,None))
    # One clip: its own renormalised probabilities and its argmax.
    if n==1:
        return p.astype(float)/p.sum(1,keepdims=True),p.argmax(1)
    # Two clips: s[a, b] = e[0, a] + e[1, b] + t[a, b], with equal labels excluded.
    if n==2:
        s=e[0,:,None]+e[1,None,:]+t
        s[np.arange(40),np.arange(40)]=-np.inf
        # best is the highest-scoring pair; q is the softmax of s over all pairs, and its row and
        # column sums are the marginals of the first and the second clip.
        best=np.asarray(np.unravel_index(s.argmax(),s.shape))
        q=np.exp(s-s.max()); q/=q.sum()
        return np.stack([q.sum(1),q.sum(0)]),best
    # Three clips: s[a, b, c] = e[0, a] + e[1, b] + e[2, c] + t[a, b] + t[b, c] over pairwise
    # distinct labels; each clip's marginal sums the softmax q over the other two axes.
    assert n==3
    s=e[0,:,None,None]+e[1,None,:,None]+e[2,None,None,:]+t[:,:,None]+t[None,:,:]
    s[~valid3]=-np.inf
    best=np.asarray(np.unravel_index(s.argmax(),s.shape))
    q=np.exp(s-s.max()); q/=q.sum()
    return np.stack([q.sum((1,2)),q.sum((0,2)),q.sum((0,1))]),best

def mpm(p,prior):
    # Takes of up to 3 clips use the exact marginals; longer takes use the final beam of a 256-beam
    # search over all 40 classes per clip. Each clip gets the argmax of its own marginal, so labels
    # may repeat within a take.
    if len(p)<=3:return full_short(p,prior)[0].argmax(1)
    return marginals_from(beams_for_take(p,prior,.75,beam_width=256,candidates_per_step=40),len(p)).argmax(1)
