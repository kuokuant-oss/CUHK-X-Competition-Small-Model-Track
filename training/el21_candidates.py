"""Verified archival candidates, frozen probabilities and full/fold pack paths."""
# Role: an index of earlier model configurations with the paths of their saved probabilities and
#   checkpoints (`registry`), and a loader for one fold's probabilities (`get_fold`); it also
#   re-exports the .npz helpers `z` and `align` of el20_scoring.
# Used by: el22_p1_heads.py (imports `z` and `align`); training. `registry` and `get_fold` are
#   not called by any script in this package.
from el21_common import ROOT
from el20_scoring import z,align
from iterative_common import populations

# {name: entry} for the earlier configurations. Each tuple holds: name, an integer reference,
# the per-fold probability file ({f} = fold number), the .npz key of those probabilities, the
# per-fold checkpoint, the full-data checkpoint and the test-set probabilities; paths are
# relative to models/. The decoder is 'MPM' for H-MPM, joint MAP otherwise.
def registry():
    entries=[
      ('F0',56185021,'fd-ensemble-20260912/F0/fold{f}/actual/probabilities.npz','fused','fd-ensemble-20260912/F0/fold{f}/deliverable-fd.pt','fd-ensemble-20260912/F0/full/deliverable-fd.pt','fd-ensemble-20260912/delivery/F0-r2/warm/probabilities.npz'),
      ('A1',56206687,'fd15-20260913/A1/fold{f}/actual/probabilities.npz','fused','fd15-20260913/A1/fold{f}/deliverable-fd15.pt','fd15-20260913/delivery/A1/model.pt','fd15-20260913/delivery/A1/warm/probabilities.npz'),
      ('X1',56209099,'fd17-20260913/actual/fold{f}/X1-control.npz','fused','fd16-20260913/actual/fold{f}/model.pt','fd16-20260913/delivery/model.pt','fd16-20260913/delivery/warm/X1/warm/probabilities.npz'),
      ('G-X1',56210960,'fd17-20260913/actual/fold{f}/G-X1.npz','fused','fd17-20260913/actual/fold{f}/model.pt','fd17-20260913/delivery/model.pt','fd17-20260913/delivery/warm/G-X1/warm/probabilities.npz'),
      ('G-X0',56210978,'fd17-20260913/actual/fold{f}/G-X0.npz','fused','fd17-20260913/actual/fold{f}/model.pt','fd17-20260913/delivery/model.pt','fd17-20260913/delivery/warm/G-X0/warm/probabilities.npz'),
      # The S1 entry points at the fused S1-G probabilities (S1 blended geometrically with F0).
      ('S1',56217468,'fd18-20260914/actual/fold{f}/S1-G.npz','fused','fd18-20260914/actual/fold{f}/model.pt','fd18-20260914/delivery/model.pt','fd18-20260914/delivery/warm/S1-G/warm/probabilities.npz'),
      ('X7',56198779,'fd13-20260913/X7/fold{f}/actual/probabilities.npz','fused','fd13-20260913/X7/fold{f}/deliverable-fd13.pt','fd13-20260913/X7/full/deliverable-fd13.pt','fd13-20260913/delivery/X7-r1/warm/probabilities.npz'),
      ('F0-thermal',56187670,'fd-ensemble-20260912/F0-plus-thermal020-MAP/fold{f}/actual/probabilities.npz','fused','fd-ensemble-20260912/F0-plus-thermal020-MAP/fold{f}/deliverable-fd.pt','fd-ensemble-20260912/F0-plus-thermal020-MAP/full/deliverable-fd.pt','fd-ensemble-20260912/delivery/F0-thermal-r1/warm/probabilities.npz'),
      ('B-H-MAP',56186141,'fd-ensemble-20260912/B-H-MAP/fold{f}/actual/probabilities.npz','fused','fd-ensemble-20260912/B-H-MAP/fold{f}/deliverable-fd.pt','fd-ensemble-20260912/B-H-MAP/full/deliverable-fd.pt','fd-ensemble-20260912/delivery/B-H-MAP-r1/warm/probabilities.npz'),
      ('H-MPM',56177611,'evening-20260911/H-MP/fold{f}/H-MP.npz','probs','evening-20260911/H-MP/fold{f}/deliverable-mp.pt','evening-20260911/H-MP-standalone/deliverable-mp.pt','evening-20260911/H-MP-standalone/hmp_0911s1_warm-probs.npz'),
      ('H-MP',56175821,'evening-20260911/H-MP/fold{f}/H-MP.npz','probs','evening-20260911/H-MP/fold{f}/deliverable-mp.pt','evening-20260911/H-MP-standalone/deliverable-mp.pt','evening-20260911/H-MP-standalone/hmp_0911s1_warm-probs.npz'),
      ('C',56162752,'abc-c-20260911/fold{f}/quant-audit/C-dual.npz','probs','abc-c-20260911/fold{f}/quant-audit/C-deliverable-int8.pt','abc-c-final-20260911/deliverable-int8.pt','abc-c-final-20260911/delivery-c_0911r2/canonical-warm-probs.npz'),
      ('T2',56158367,'abc-interp-20260911/fold{f}/T2/quant-audit/after.npz','probs','abc-interp-20260911/fold{f}/T2/quant-audit/deliverable.pt','abc-t2-final-20260911/deliverable-int8.pt','abc-t2-final-20260911/delivery-t2_0911r1/warm-probs.npz'),
      ('BNcal',56157727,'abc-interp-20260911/fold{f}/B2-BNcal/quant-audit/after.npz','probs','abc-interp-20260911/fold{f}/B2-BNcal/quant-audit/deliverable.pt','abc-b2-bncal-final-20260911/deliverable-int8.pt','abc-b2-bncal-final-20260911/delivery-bncal_0911r1/warm-probs.npz'),
      ('B2',56156407,'abc-b-20260911/fold{f}/B2/quant-audit/after.npz','probs','abc-b-20260911/fold{f}/B2/quant-audit/deliverable.pt','abc-b2-final-20260911/deliverable-int8.pt','abc-b2-final-20260911/delivery-b2_0911r1/warm-probs.npz')]
    return {a:dict(arm=a,reference=r,fold_probabilities=p,probability_key=k,fold_pack=fp,full_pack=full,test_probabilities=test,decoder='MPM' if a=='H-MPM' else 'MAP') for a,r,p,k,fp,full,test in entries}

# Fold f: its clip ids, the probabilities of each requested configuration (all by default)
# aligned to those ids, and the list of files read.
def get_fold(f,arms=None):
    t,_,_=populations();ids=list(t[t.fold==f].index);regs=registry();arrays={};paths=[]
    for a in arms or regs:
        v=regs[a];p=ROOT/'models'/v['fold_probabilities'].format(f=f);arrays[a]=align(z(p),ids,v['probability_key']);paths.append(p)
    return ids,arrays,paths
