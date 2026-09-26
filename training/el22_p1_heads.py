"""All four fixed P1 S1-form heads plus one clean early-only head; P0 replay."""
# Role: fits the readout head of family S1 (logistic regression on [I, D, I⊙D, |I−D|] built from
#   P1 region features). With --full it fits the delivered head on all 2,931 training clips;
#   without it, one head per validation fold plus a fifth validation split, `early_to_late`.
# Used by: run directly (`python training/el22_p1_heads.py [--full]`); training.
# Main steps: 1) check the feature files against their recorded SHA256 2) load the P1 and P0
#   region features 3) per job: for validation jobs, replay the earlier P0 head bit for bit;
#   fit the P1 head, check that the saved head reloads to identical probabilities, and write
#   head.pt, predictions.npz and a JSON record with the SHA256 of every output and source.
# el22_common is imported first so that its side effects precede numpy: it puts src/, scripts/
# and training/ on sys.path and sets single-threaded BLAS. Its star import also supplies Path,
# ROOT, DEST and the helpers j (read JSON), sha (file SHA256), write (atomic JSON write), now
# and sources.
from el22_common import *
import numpy as np,pandas as pd,torch,argparse,time
# features(): transformer features of the 2,931 training clips and the clip table (index = clip
# id; columns include fold, user and action_id), both in the same clip order.
from fd17_heads import features
# fit(): per-modality standardisation, design matrix and logistic-regression fit of one head.
from fd18_heads import fit
# predict(): the readout as computed at inference; (clips, 40) probabilities, mean of both flips.
from cuhkx.fd18_readout import predict
# z(): load an .npz file into a dict of arrays (align is imported but not used here).
from el21_candidates import z,align

def main(full=False):
    # Step 1: inputs. Torch runs single-threaded; `base` = DEST/'P1' is this head's working
    # folder; policy.json holds the clip order and each job's fit and validation clip ids; the
    # JSON record of the feature-extraction step must report success.
    torch.set_num_threads(1);base=DEST/'P1';policy=j(base/'policy.json');receipt=j(base/'features/receipt.json');assert receipt['passed']
    # Every feature file must still have the SHA256 recorded when it was written.
    for p,h in receipt['output_sha256'].items():assert sha(p)==h
    # Load the transformer features and the clip table; `original` is a separate copy of the
    # feature dict that receives the P0 features below; the clip order must match policy.json.
    x,table=features();original=dict(x);ids=list(table.index);assert ids==policy['train_ids']
    # Step 2: region features per modality (I = IR, D = depth), shape (clips, 2 flips, 5 regions,
    # 384), memory-mapped: `x` gets the P1 (support-weighted) pooling from this folder and
    # `original` the earlier P0 (unweighted) pooling.
    for mod in ['I','D']:
        x['S'+mod]=np.load(base/f'features/S{mod}.npy',mmap_mode='r');original['S'+mod]=np.load(ROOT/f'models/fd18-20260914/features/S{mod}.npy',mmap_mode='r')
    # True where a training clip has both IR and depth; saved with every prediction file as
    # `available` (at inference the readout is blended into F0 only on such clips).
    cov=pd.read_csv(ROOT/'models/el21-validation-20260914/V2a/covariates.csv').set_index('clip_id')
    alive=cov.loc[ids,'IR_available'].to_numpy()&cov.loc[ids,'Depth_available'].to_numpy()
    # Default jobs: one head per validation fold, fitted with that fold held out (METHOD.md
    # section 7), plus the split `early_to_late`; all clip lists come from policy.json.
    jobs=['fold0','fold1','fold2','fold3','early_to_late']
    if full:
        # --full refuses to run unless the fold-scoring decision (scoring/decision.json) approves
        # the P1 head for delivery; it then fits a single head on all training clips.
        assert j(base/'scoring/decision.json')['qualified_for_delivery'];jobs=['full']
    for name in jobs:
        out=base/'heads'/name
        # A job that already has its completion record is not refitted: its outputs are checked
        # against the recorded SHA256 values and reused.
        if (out/'receipt.json').exists():
            for p,h in j(out/'receipt.json')['output_sha256'].items():assert sha(p)==h
            print(dict(head=name,reused=True),flush=True);continue
        # New job: its output folder must not exist yet; start the timer.
        out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
        # Fit clips: all training clips for --full (no validation clips), otherwise the job's
        # fit and validation lists from policy.json.
        train=ids if full else policy['fit_ids'][name];valid=[] if full else policy['validation_ids'][name]
        # Table row of every fit clip; each must exist and none may also be a validation clip.
        rr=table.index.get_indexer(train);assert (rr>=0).all() and not set(train)&set(valid)
        # Step 3 (validation jobs only): P0 replay. The earlier S1 head of the same split, fitted
        # on P0 features, must reproduce its saved probabilities exactly; its probabilities for
        # all clips are then saved as P0-predictions.npz next to the new head's predictions.
        if not full:
            # Earlier head: the fold job's prototype head, or the early_to_late validation head.
            hp=ROOT/f'models/fd18-20260914/prototype/{name}/S1-head.pt' if name.startswith('fold') else ROOT/'models/el21-validation-20260914/external/batch-panel/early_to_late/S1/head.pt'
            # Load it; it must have been fitted on exactly this job's fit clips.
            h0=torch.load(hp,map_location='cpu',weights_only=False);assert set(h0['train_ids'])==set(train)
            # Its probabilities for all 2,931 clips, computed from the P0 features.
            q0=predict(original,h0)
            # The probabilities saved with that head (fold jobs: S1-outer.npz).
            old=ROOT/f'models/fd18-20260914/prototype/{name}/S1-outer.npz' if name.startswith('fold') else hp.parent/'predictions.npz'
            # Match the saved rows by clip id and require bit-for-bit equality with the replay.
            v=z(old);ii=table.index.get_indexer(list(v['clip_ids']));assert np.array_equal(q0[ii],v['probs'])
            np.savez_compressed(out/'P0-predictions.npz',clip_ids=np.array(ids,dtype=str),probs=q0,available=alive)
        # Step 4: fit the S1 head on the P1 features of the fit clips. q = its probabilities for
        # all 2,931 clips, shape (clips, 40); di = fit diagnostics; C must be 0.01.
        head,q,di=fit(x,table,rr,'S1');assert head['C']==.01 and set(head['train_ids'])==set(train)
        # Save the head, reload it with the restricted loader (weights_only=True: tensors and plain
        # Python values only) and require that the reloaded head gives exactly q.
        torch.save(head,out/'head.pt');loaded=torch.load(out/'head.pt',map_location='cpu',weights_only=True);actual=predict(x,loaded);assert np.array_equal(actual,q)
        # Probabilities of all 2,931 clips, with the availability flag.
        np.savez_compressed(out/'predictions.npz',clip_ids=np.array(ids,dtype=str),probs=actual,available=alive)
        # Completion record: time, job, fit and validation clip ids, flags for the two exactness
        # checks (the P0 replay flag is False for --full, where no replay runs), fit diagnostics,
        # run time, and the SHA256 of the outputs and of the sources (this script, policy.json,
        # the feature record, fd18_heads.py, fd18_readout.py and, for validation jobs, the earlier
        # head and its saved probabilities).
        write(out/'receipt.json',dict(at=now(),head=name,passed=True,fit_ids=train,validation_ids=valid,actual_FP32_head_reload_exact=True,P0_original_head_replay_exact=not full,fit=di,seconds=time.perf_counter()-start,output_sha256=sources([out/'head.pt',out/'predictions.npz',*([out/'P0-predictions.npz'] if not full else [])]),source_sha256=sources([Path(__file__),base/'policy.json',base/'features/receipt.json',ROOT/'training/fd18_heads.py',ROOT/'src/cuhkx/fd18_readout.py',*([hp,old] if not full else [])])))
        print(dict(head=name,complete=True,seconds=time.perf_counter()-start,P0_exact=not full),flush=True)

if __name__=='__main__':
    # Command line: --full fits the delivered head on all training clips; default: validation heads.
    ap=argparse.ArgumentParser();ap.add_argument('--full',action='store_true');main(ap.parse_args().full)
