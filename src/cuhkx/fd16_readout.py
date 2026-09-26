"""FD16 deterministic component composition shared by runtime and verification."""
# Role: diagnostic masking of modalities (MASKS), rebuilding C and F0 from the member outputs
#   (base_components), and the readout path for heads on whole-grid or temporal features.
# Used by: fd18_readout (MASKS, base_components, predict) and training/fd17_heads.py; both.
import numpy as np
from cuhkx.fd15_codec import predict_head

# Diagnostic phases; scripts/el25r_p1_repeat_runtime.py writes one prediction per phase, and
# 'warm' (nothing hidden) gives the submission. 'Ipartial'/'Iall' hide IR on every 7th clip / on
# all clips, 'Dpartial'/'Dall' depth, 'localpartial'/'localall' member C's miw view; 'newall'
# hides IR and depth (so no readout is blended), 'allmissing' hides all three.
MASKS=['warm','Ipartial','Iall','Dpartial','Dall','localpartial','localall','newall','allmissing']

# Applies one mask mode to a presence vector: 'all' clears it, 'partial' clears clips 0, 7, 14, ...
# (position divisible by 7), 'none' leaves it unchanged.
def masked(present,kind):
    x=np.asarray(present,dtype=bool).copy()
    if kind=='all':x[:]=False
    elif kind=='partial':x[np.arange(len(x))%7==0]=False
    else:assert kind=='none'
    return x

# The mask mode ('none', 'partial' or 'all') of a phase for 'local' (the miw view), 'I' and 'D'.
def mask_modes(phase):
    return {k:('partial' if phase==k+'partial' else 'all' if phase in [k+'all','allmissing'] or (k in ['I','D'] and phase=='newall') else 'none') for k in ['local','I','D']}

# Copies every component, applies the phase's masks and rebuilds C and F0 (the miw fallback):
# C is member C's det/miw average where the miw view is present and C on det alone (C_det)
# elsewhere; F0 = 1/2 B + 1/2 C in FP64. Under 'warm' this is the F0 of fd13_inference.infer_f0.
def base_components(v,phase):
    modes=mask_modes(phase);out={k:np.asarray(x).copy() for k,x in v.items()}
    for k in ['local','I','D']:out[k+'_present']=masked(v[k+'_present'],modes[k])
    out['C']=np.where(out['local_present'][:,None],v['C'],v['C_det'])
    out['F0']=.5*(out['B'].astype(np.float64)+out['C'].astype(np.float64))
    return out

# Design for the whole-grid and temporal heads below: per-modality standardisation, then [I, D]
# (layout X0) or [I, D, I*D, |I-D|] (X1). Not used by the delivered S1 head.
def joint(features,scalers,layout='X1'):
    # Installed sklearn explicitly casts stored FP64 parameters to input FP32,
    # then performs separate subtract/divide. Retain the FP64 learned payload.
    vals=[]
    for k in ['I','D']:
        x=np.array(features[k],dtype=np.float32,copy=True)
        x-=np.asarray(scalers[k]['mean'],dtype=np.float32);x/=np.asarray(scalers[k]['scale'],dtype=np.float32);vals.append(x)
    a,b=vals
    return np.concatenate([a,b] if layout=='X0' else [a,b,a*b,abs(a-b)],axis=-1)

# Class probabilities (clips, 40) for the head types 'global_joint' (whole-grid I and D),
# 'temporal_I' and 'temporal_joint', averaged over the two views. Not used by the delivered
# S1 head.
def predict(features,spec):
    typ=spec['readout_type']
    if typ=='global_joint':x=joint({k:features[k] for k in ['I','D']},spec['modality_scalers'],spec.get('layout','X1'))
    elif typ=='temporal_I':x=features['TI'].reshape(len(features['TI']),2,-1)
    elif typ=='temporal_joint':
        x=joint({k:features['T'+k] for k in ['I','D']},spec['modality_scalers']).reshape(len(features['TI']),2,-1)
    else:raise ValueError(typ)
    return predict_head(x,spec['head']).mean(1).numpy()

# Arithmetic fusion 0.8 F0 + 0.2 v for this module's configurations (v: the named readout, or
# for 'M' the mean of the 'X1' and 'TX' readouts; 'TI' requires IR only). Not called by the
# delivered run, which uses fd18_readout.compose.
def compose(components,readouts,candidate,phase='warm'):
    out=base_components(components,phase)
    if candidate=='TI':alive=out['I_present']
    else:alive=out['I_present']&out['D_present']
    v=.5*np.asarray(readouts['X1'],dtype=np.float64)+.5*np.asarray(readouts['TX'],dtype=np.float64) if candidate=='M' else readouts[candidate]
    fused=out['F0'].copy();fused[alive]=.8*out['F0'][alive]+.2*np.asarray(v,dtype=np.float64)[alive]
    out.update(member=np.asarray(v),fused=fused,new_present=alive)
    return out
