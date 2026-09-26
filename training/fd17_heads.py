"""Fixed FD16 readouts; no global state writes to earlier campaigns."""
# Role: loads the transformer features of the training clips together with the clip table
#   (`features`), and fits the earlier readout heads on whole-grid or temporal features (`fit`).
# Used by: el22_p1_heads.py (`features`) and fd18_heads.py (`fit`, for the families X1 and XR);
#   training.
import argparse,warnings,time
from importlib import import_module
# Star import: ROOT, sys, sha, j and populations from fd17_common (which also extends sys.path).
from fd17_common import *
# OLD: folder of an earlier development iteration that holds the features file. Its
# source/cpu-libs folder goes first on sys.path, so a module found there is imported in
# preference to the installed one, for every later import in the process.
OLD=ROOT/'models/fd15-20260913'
sys.path.insert(0,str(OLD/'source/cpu-libs'))

# Returns (x, table): x = the arrays of LP.npz by name ('clip_ids'; 'I' and 'D' = whole-grid
# features of IR and depth), table = the clip table from `populations`, in the same clip order.
# temporal=True also adds the temporal features 'TI' and 'TD'.
def features(temporal=False):
    import numpy as np
    from fd17_scoring import z
    # The features file is pinned by its SHA256.
    p=OLD/'features/LP.npz';assert sha(p)=='ee325a362c17c2bf5fc90a77b432474b652418468e3bfe4a77d892057be6fb9d'
    # Load it and the clip table; the clip order must be identical.
    x=z(p);table,_,_=populations();assert list(x['clip_ids'])==list(table.index)
    if temporal:
        # The record of the temporal-feature extraction must report success, and each file must
        # match the SHA256 listed there.
        r=j(ROOT/'models/fd16-20260913/features/temporal-receipt-r2.json');assert r['passed']
        for k in ['I','D']:
            p=ROOT/f'models/fd16-20260913/features/T{k}-r2.npy';assert sha(p)==r['files'][k]['sha256'];x['T'+k]=np.load(p,allow_pickle=False)
    return x,table

# Fits one earlier readout head on the table rows `rows`. cid selects the design: 'TI' = temporal
# IR features alone; 'TX' = temporal [I, D, I⊙D, |I−D|]; any other = whole-grid features, laid
# out as [I, D] for 'X0' or [I, D, I⊙D, |I−D|] otherwise. C = 1 for X0, X1 and J, else 0.01.
# Features have shape (clips, 2, ...); for the whole-grid features axis 1 holds the plain and the
# flipped encoding. Returns (spec, q, diagnostics): spec is the head as saved, q the (clips, 40)
# probabilities of all clips, averaged over axis 1.
def fit(x,table,rows,cid):
    import numpy as np,torch
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    from cuhkx.fd16_readout import joint,predict
    # Head specification: readout type, layout, fit clip ids and regularisation C.
    typ='temporal_I' if cid=='TI' else 'temporal_joint' if cid=='TX' else 'global_joint';spec=dict(readout_type=typ,layout='X0' if cid=='X0' else 'X1',train_ids=list(table.index[rows]),C=1. if cid in ['X0','X1','J'] else .01)
    # 'TI': the temporal IR features are used as they are, reshaped to (clips, 2, dim), with no
    # per-modality standardisation.
    if cid=='TI':z=x['TI'].reshape(len(table),2,-1);spec['modality_scalers']={}
    else:
        # Otherwise, per modality: a per-dimension StandardScaler fitted on the fit clips (both
        # entries of axis 1), stored in the spec; then the design of every clip, (clips, 2, dim).
        inp={k:x[('T' if cid=='TX' else '')+k] for k in ['I','D']};scalers={}
        for k in inp:
            shape=inp[k].shape[2:];sc=StandardScaler().fit(inp[k][rows].reshape(len(rows)*2,-1));scalers[k]=dict(mean=torch.tensor(sc.mean_.reshape(shape)),scale=torch.tensor(sc.scale_.reshape(shape)))
        spec['modality_scalers']=scalers;z=joint(inp,scalers,spec['layout']).reshape(len(table),2,-1)
    # Second standardisation, fitted on the fit rows; each clip gives two rows (axis 1) with its
    # label repeated.
    dim=z.shape[-1];sc=StandardScaler().fit(z[rows].reshape(-1,dim));xx=sc.transform(z[rows].reshape(-1,dim));yy=np.repeat(table.action_id.to_numpy()[rows],2);attempts=[]
    # L-BFGS multinomial logistic regression, each row at sample weight 0.5; if 1,000 iterations
    # raise a ConvergenceWarning, the fit is repeated with 3,000.
    for limit in [1000,3000]:
        m=LogisticRegression(C=spec['C'],solver='lbfgs',max_iter=limit,tol=1e-4,class_weight=None)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always');m.fit(xx,yy,sample_weight=np.full(len(yy),.5))
        failed=any(issubclass(w.category,ConvergenceWarning) for w in ws);attempts.append(dict(limit=limit,iterations=m.n_iter_.tolist(),converged=not failed))
        if not failed:break
    assert not failed,'Fixed solver did not converge'
    # Deployed head: FP32 mean and scale of the second standardisation, weights and bias, plus
    # the class ids (int64) that map the head's outputs to the 40 actions.
    h={k:torch.tensor(v,dtype=torch.float32) for k,v in dict(mean=sc.mean_,scale=sc.scale_,weight=m.coef_,bias=m.intercept_).items()};h['classes']=torch.tensor(m.classes_,dtype=torch.int64);spec['head']=h
    from cuhkx.fd15_codec import predict_head
    # The deployed FP32 computation must reproduce scikit-learn's probabilities on the fit rows
    # (atol 2e-6, rtol 2e-4), and the readout's predict must equal the mean over axis 1 exactly.
    pq=predict_head(z,h).numpy();native=m.predict_proba(xx);assert np.allclose(native,pq[rows].reshape(-1,40)[:,m.classes_],atol=2e-6,rtol=2e-4)
    q=predict(x,spec);assert np.array_equal(q,pq.mean(1))
    # Diagnostics: solver attempts, dimension, the largest scikit-learn/deployed difference, the
    # fit subjects, and accuracy, NLL (probabilities clipped at 1e-12) and mean top probability
    # on the fit clips ('clean') and on all other clips ('outer').
    y=table.action_id.to_numpy();valid=np.setdiff1d(np.arange(len(table)),rows)
    def stats(ii):return dict(n=len(ii),accuracy=float((q[ii].argmax(1)==y[ii]).mean()),nll=float(-np.log(np.clip(q[ii,y[ii]],1e-12,1)).mean()),confidence=float(q[ii].max(1).mean())) if len(ii) else None
    return spec,q,dict(attempts=attempts,dimension=dim,solver_deployed_max_abs=float(abs(native-pq[rows].reshape(-1,40)[:,m.classes_]).max()),clean=stats(rows),outer=stats(valid),train_subjects=sorted(set(map(str,table.user.iloc[rows]))))

