"""Training-only fixed FD18 heads and numeric round-robin HC groups."""
# Role: fits readout heads by family. For the delivered family S1: standardise the P1 region
#   features per modality, build [I, D, I⊙D, |I−D|] (7,680 dimensions), standardise again and
#   fit an L2 logistic regression with C = 0.01. Also metrics and a temperature-calibrated bag
#   of four inner heads (`metric`, `groups`, `bag`).
# Used by: el22_p1_heads.py (`fit` with family S1); training. `bag` (with `groups`, `metric`)
#   is not called by any script in this package.
import warnings,time
# Star import: Path, sha, write and the other helpers of fd18_common (which also extends sys.path).
from fd18_common import *
from fd17_heads import features,fit as legacy_fit

# Fits one head of `family` on the table rows `rows`. x holds the features by name: 'SI', 'SD' =
# region features (clips, 2 flips, 5 regions, 384), read by S0, S1 and SN; the other families
# read 'I' and 'D'. Returns (spec, q, diagnostics): spec is the head as saved (head.pt), q the
# (clips, 40) probabilities of all clips, averaged over the two flips.
def fit(x,table,rows,family):
    import numpy as np,torch
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    from cuhkx.fd18_readout import design,predict
    # X1 and XR are fitted by the earlier fitter, fd17_heads.fit (a joint design on 'I' and 'D').
    if family in ['X1','XR']:return legacy_fit(x,table,rows,family)
    # Head specification: C = 1 for N0, T0 and C0, else 0.01 (so 0.01 for S1).
    spec=dict(readout_type='fd18_fixed',family=family,C=1. if family in ['N0','T0','C0'] else .01,train_ids=list(table.index[rows]),centers={},modality_scalers={})
    # Step 1, per modality (I = IR, D = depth): region features for S0, S1 and SN, 'I'/'D'
    # otherwise; `shape` is the per-flip feature shape, (5, 384) for region features.
    for k in ['I','D']:
        v=x[('S' if family in ['S0','S1','SN'] else '')+k];shape=v.shape[2:]
        # N0 and SN store the mean feature of the fit clips as a centre (fd18_readout.direction
        # then uses directions from it); every other family, S1 included, stores a
        # per-dimension StandardScaler fitted on the fit clips, both flips.
        if family in ['N0','SN']:spec['centers'][k]=torch.tensor(v[rows].mean((0,1)),dtype=torch.float32)
        else:
            sc=StandardScaler().fit(v[rows].reshape(len(rows)*2,-1));spec['modality_scalers'][k]=dict(mean=torch.tensor(sc.mean_.reshape(shape)),scale=torch.tensor(sc.scale_.reshape(shape)))
    # Step 2: design of every clip, (clips, 2 flips, dim) with dim = 7,680 for S1; a second
    # StandardScaler fitted on the fit rows; each clip gives two rows with its label repeated.
    z=design(x,spec);dim=z.shape[-1];sc=StandardScaler().fit(z[rows].reshape(-1,dim));xx=sc.transform(z[rows].reshape(-1,dim));y=table.action_id.to_numpy();yy=np.repeat(y[rows],2);attempts=[]
    # Step 3: L-BFGS multinomial logistic regression, each flip at sample weight 0.5; if 1,000
    # iterations raise a ConvergenceWarning, the fit is repeated with 3,000.
    for limit in [1000,3000]:
        m=LogisticRegression(C=spec['C'],solver='lbfgs',max_iter=limit,tol=1e-4,class_weight=None)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always');m.fit(xx,yy,sample_weight=np.full(len(yy),.5))
        failed=any(issubclass(w.category,ConvergenceWarning) for w in ws);attempts.append(dict(limit=limit,iterations=m.n_iter_.tolist(),converged=not failed))
        if not failed:break
    assert not failed
    # Step 4: the deployed head: FP32 mean and scale of the second standardisation, weights and
    # bias, plus the class ids (int64) that map the head's outputs to the 40 actions.
    spec['head']={k:torch.tensor(v,dtype=torch.float32) for k,v in dict(mean=sc.mean_,scale=sc.scale_,weight=m.coef_,bias=m.intercept_).items()};spec['head']['classes']=torch.tensor(m.classes_,dtype=torch.int64)
    from cuhkx.fd15_codec import predict_head
    # The deployed FP32 computation must reproduce scikit-learn's probabilities on the fit rows
    # (atol 2e-6, rtol 2e-4).
    native=m.predict_proba(xx);deployed=predict_head(z[rows],spec['head']).numpy().reshape(-1,40)[:,m.classes_];assert np.allclose(native,deployed,atol=2e-6,rtol=2e-4)
    # Probabilities of all clips through the inference readout; diagnostics: solver attempts,
    # dimension, C, largest scikit-learn/deployed difference, and the classes absent from the fit
    # clips (the head gives them probability 0).
    q=predict(x,spec);return spec,q,dict(attempts=attempts,dimension=dim,C=spec['C'],solver_deployed_max_abs=float(abs(native-deployed).max()),missing_train_classes=sorted(set(range(40))-set(m.classes_.tolist())),train_ids=spec['train_ids'])

# Accuracy, NLL (probabilities clipped at 1e-12), Brier score (squared distance to the one-hot
# label) and mean top probability of probabilities q (clips, 40) against labels y.
def metric(q,y):
    import numpy as np
    return dict(n=len(y),accuracy=float((q.argmax(1)==y).mean()),nll=float(-np.log(np.clip(q[np.arange(len(y)),y],1e-12,1)).mean()),brier=float((np.square(q).sum(1)-2*q[np.arange(len(y)),y]+1).mean()),confidence=float(q.max(1).mean()))

# Assigns each user to one of four inner groups: users numbered 9 or lower first, then the rest,
# each cohort sorted by number and dealt round-robin (0, 1, 2, 3, 0, ...), the count carrying
# over from the first cohort to the second.
def groups(users):
    import re
    number=lambda u:int(re.search(r'\d+',u).group());out={};offset=0
    for low in [True,False]:
        cohort=sorted({str(u) for u in users if (number(str(u))<=9)==low},key=number)
        out.update({u:(i+offset)%4 for i,u in enumerate(cohort)});offset+=len(cohort)
    return out

# Bag of four inner X1 heads on the fit rows: inner head k is fitted without group k's users and
# its temperature T is chosen on group k's clips by minimising NLL over T in [0.25, 4]. Saves the
# inner heads, HB (plain mean of their probabilities), HC (mean of the temperature-scaled
# probabilities) and a JSON record; returns both bags and their probabilities for all clips.
def bag(x,table,rows,folder):
    import numpy as np,torch
    from scipy.optimize import minimize_scalar
    from cuhkx.fd18_readout import temperature,predict
    # New output folder; the user of every clip; the inner group of each fit-clip user; labels.
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False);us=table.user.astype(str).to_numpy();g=groups(us[rows]);y=table.action_id.to_numpy();members=[];receipts=[]
    for k in range(4):
        # Inner split: fit rows outside group k train the head, group k calibrates it; no user
        # is in both.
        tr=np.array([i for i in rows if g[us[i]]!=k]);cal=np.array([i for i in rows if g[us[i]]==k]);assert not set(us[tr])&set(us[cal]);h,q,di=fit(x,table,tr,'X1')
        # Bounded search over log T (tolerance 1e-4); save the inner head with its temperature.
        res=minimize_scalar(lambda logT:metric(temperature(q[cal],np.exp(logT)),y[cal])['nll'],bounds=(np.log(.25),np.log(4)),method='bounded',options={'xatol':1e-4});assert res.success;T=float(np.exp(res.x));p=folder/f'inner{k}-head.pt';torch.save(h,p);members.append(dict(spec=h,temperature=T))
        # Record of the inner head: clip ids, T (flagged when at or next to a search bound), classes
        # absent from its fit clips, head SHA256, fit diagnostics, and metrics on group k before
        # and after temperature scaling.
        receipts.append(dict(inner=k,train_ids=list(table.index[tr]),calibration_ids=list(table.index[cal]),temperature=T,temperature_boundary=T<.251 or T>3.99,missing_train_classes=sorted(set(range(40))-set(h['head']['classes'].tolist())),head_sha256=sha(p),fit=di,calibration_before=metric(q[cal],y[cal]),calibration_after=metric(temperature(q[cal],T),y[cal])))
    # HB and HC share the four inner heads; only HC applies their temperatures when predicting.
    hb=dict(readout_type='bag',family='HB',calibrated=False,members=members,train_ids=list(table.index[rows]),groups=g);hc=dict(hb,family='HC',calibrated=True)
    # Save both bags and the JSON record (groups, inner-head records, fit clip ids); return the
    # bags and their probabilities for all clips.
    torch.save(hc,folder/'HC-head.pt');torch.save(hb,folder/'HB-head.pt');write(folder/'receipt.json',dict(passed=True,groups=g,heads=receipts,train_ids=list(table.index[rows])));return {'HC':hc,'HB':hb},{'HC':predict(x,hc),'HB':predict(x,hb)}
