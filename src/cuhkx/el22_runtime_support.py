"""Support geometry from actual raw dimensions and actual detector windows."""
# Role: computes each clip's in-frame support weights for P1 pooling from its raw frame size and
#   its person window.
# Used by: el22_support_inference.components; inference.
import numpy as np
import pandas as pd
from PIL import Image
from cuhkx.paths import p
from cuhkx.el22_support_pool import patch_support

# Returns (clips, 2, 14, 14) float32 in the order of `ids`: index 0 for the plain view, 1 for the
# horizontally flipped view. `split` ('train' or 'test') selects where the raw frames and the
# person windows are read.
def support_for_ids(ids,split):
    if split not in ['train','test']:raise ValueError(split)
    # Person windows written by scripts/22_person_windows.py, one row per clip.
    windows=pd.read_parquet(p('processed_root')/f'person_windows_{split}.parquet').set_index('clip_id')
    masks=[]
    for c in ids:
        # The clip's raw Depth_Color folder; the size of its first frame is the frame size on
        # which the window was cut.
        folder=p('train_root')/'Depth_Color'/c if split=='train' else p('test_root')/c/'Depth_Color'
        paths=sorted(folder.glob('*.png'));assert paths,('No raw Depth shape source',c)
        with Image.open(paths[0]) as image:width,height=image.size
        window=windows.loc[c,['top','left','bottom','right']].to_numpy()
        masks.append(np.stack([patch_support(height,width,window,bool(v)) for v in [0,1]]))
    return np.stack(masks)
