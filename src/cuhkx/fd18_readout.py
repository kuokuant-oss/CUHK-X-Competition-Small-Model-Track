"""Fixed FD18 feature axes, per-head calibration and original fallback."""
# Role: readout heads on the pooled ViT features (design matrix and prediction) and the fusion of
#   a readout with F0 (compose): arithmetic (-A) or geometric (-G) blend at weight 0.2 on clips
#   with both IR and depth.
# Used by: el22_support_inference and scripts/el25r_p1_repeat_runtime.py (inference),
#   fd18_codec (graph), training/fd18_heads.py and training/el22_p1_heads.py; both.
import numpy as np
from cuhkx.fd15_codec import predict_head
from cuhkx.fd16_readout import MASKS,masked,mask_modes,base_components,predict as old_predict

# Direction features of the N0 and SN families: subtract a stored centre and rescale each
# 384-dim vector to length sqrt(384). Not used by the delivered S1 head.
def direction(x,center):
    v=np.array(x,dtype=np.float32,copy=True)-np.asarray(center,dtype=np.float32)
    norm=np.linalg.norm(v,axis=-1,keepdims=True)
    return v/np.maximum(norm,np.float32(1e-12))*np.float32(np.sqrt(384))

# Head input, (clips, 2 views, n). For S1: the (clips, 2, 5, 384) region features of each
# modality are standardised, then [I, D, I*D, |I-D|] is concatenated per region and flattened,
# giving 5 x 4 x 384 = 7,680 values ordered region by region.
def design(features,spec):
    family=spec['family'];vals=[]
    for mod in ['I','D']:
        # Region families (S0, S1, SN) read 'SI'/'SD'; the others the whole-grid 'I'/'D'.
        raw=features[('S' if family in ['S0','S1','SN'] else '')+mod]
        if family in ['N0','SN']:v=direction(raw,spec['centers'][mod])
        else:
            # Per-modality standardisation with the training mean and scale, stored in FP64 and
            # applied in FP32 (subtract, then divide).
            v=np.array(raw,dtype=np.float32,copy=True);sc=spec['modality_scalers'][mod]
            v-=np.asarray(sc['mean'],np.float32);v/=np.asarray(sc['scale'],np.float32)
        vals.append(v)
    # S0, SN and G0c use [I, D] only; the other families, S1 included, add the elementwise
    # product and the absolute difference.
    a,b=vals;parts=[a,b] if family in ['S0','SN','G0c'] else [a,b,a*b,abs(a-b)]
    return np.concatenate(parts,axis=-1).reshape(len(a),2,-1)

# Temperature scaling of probabilities: softmax(log q / T), with q clipped at 1e-12.
def temperature(q,T):
    z=np.log(np.clip(np.asarray(q,np.float64),1e-12,1))/T;z-=z.max(1,keepdims=True);p=np.exp(z);return p/p.sum(1,keepdims=True)

# Class probabilities (clips, 40) of one head. 'bag' averages several heads, each predicted by
# fd16_readout.predict (and temperature-scaled first if calibrated); 'global_joint' uses
# fd16_readout.predict directly. The delivered S1 head takes the last line: predict_head on the
# design matrix, averaged over the two views.
def predict(features,spec):
    if spec['readout_type']=='bag':
        qs=[old_predict(features,h['spec']) for h in spec['members']]
        return np.mean(np.stack([temperature(q,h['temperature']) for q,h in zip(qs,spec['members'])]),0) if spec['calibrated'] else np.mean(np.stack(qs),0).astype(np.float64)
    if spec['readout_type']=='global_joint':return old_predict(features,spec)
    return predict_head(design(features,spec),spec['head']).mean(1).numpy()

# Fuses F0 with one readout. `candidate` is '<head family>-<A|G>', e.g. 'S1-G'; `phase` is one of
# the diagnostic phases in fd16_readout.MASKS ('warm' hides nothing).
def compose(base,readouts,candidate,phase='warm'):
    # In order: copy the components with the phase's masks applied and C and F0 rebuilt; split
    # the name into head family and blend kind; the readout q in FP64; `alive` marks the clips
    # with both IR and depth present, the only clips whose F0 is changed.
    out=base_components(base,phase);member,kind=candidate.rsplit('-',1);q=np.asarray(readouts[member],np.float64);alive=out['I_present']&out['D_present'];f=out['F0'].copy()
    # Arithmetic blend: 0.8 F0 + 0.2 q.
    if kind=='A':f[alive]=.8*f[alive]+.2*q[alive]
    else:
        # Geometric blend: exp(0.8 log F0 + 0.2 log q) with both clipped at 1e-12; the row
        # maximum is subtracted for numerical stability before exp, then each row is normalised.
        assert kind=='G';z=.8*np.log(np.clip(f[alive],1e-12,1))+.2*np.log(np.clip(q[alive],1e-12,1));z-=z.max(1,keepdims=True);v=np.exp(z);f[alive]=v/v.sum(1,keepdims=True)
    # 'member': the readout; 'new_present': the blended clips; 'fused': the final probabilities.
    out.update(member=q,new_present=alive,fused=f);return out

# Description of how a configuration is composed. The checkpoint stores it for every selected
# configuration and fd18_codec.load checks it against this function. Here 'pooling' holds the
# blend kind (A or G), not the P0/P1 region pooling.
def graph(candidate):
    family,kind=candidate.rsplit('-',1)
    return dict(family=family,pooling=kind,alpha=.2,alive='I_AND_D',dead='current_F0',feature='regions' if family in ['S0','S1','SN'] else 'raw_TSN' if family in ['T0','C0'] else 'global')
