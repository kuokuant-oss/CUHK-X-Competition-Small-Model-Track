"""P0 exact original pooling; P1 geometric support-weighted final pooling."""
# Role: region pooling of the last-block ViT tokens. P0 averages each region plainly; P1 weights
#   each 16x16 patch by the fraction of its pixels that lie inside the camera frame.
# Used by: el22_support_inference (extract_pair), el22_runtime_support (patch_support) and
#   tools/make_preprocessing_figure.py (patch_support); inference.
import numpy as np
from PIL import Image
import torch
from cuhkx.fd15_vit import prefix

# The five pooling regions on the 14x14 patch grid as (row start, row end, column start, column
# end), ends exclusive: the whole grid, then the top-left, top-right, bottom-left and
# bottom-right 7x7 quadrants.
REGIONS=[(0,14,0,14),(0,7,0,7),(0,7,7,14),(7,14,0,7),(7,14,7,14)]

# Support weights of one clip, (14, 14) float32 in [0, 1]: the fraction of each 16x16 patch of the
# centre 224 crop that shows camera pixels rather than repeated edge pixels. `window` is the
# person window (top, left, bottom, right) in raw-frame pixels and may extend beyond the
# height x width frame; flip=True gives the weights for the horizontally flipped view.
def patch_support(height,width,window,flip=False):
    top,left,bottom,right=map(int,window)
    if height<=0 or width<=0 or bottom<=top or right<=left:raise ValueError('Invalid source/crop geometry')
    # Same PIL bilinear resize geometry as fused.load_clip_fused, without uint8
    # rounding: support is a fractional pixel weight, not an image channel.
    # Window-sized mask: 1 where the window pixel lies inside the frame, 0 outside.
    yy=np.arange(top,bottom);xx=np.arange(left,right)
    support=((yy[:,None]>=0)&(yy[:,None]<height)&(xx[None,:]>=0)&(xx[None,:]<width)).astype(np.float32)
    # Resize to the det248 size as a float image, then take the centre 224 crop that
    # fd15_vit.preprocess takes.
    m=np.asarray(Image.fromarray(support).resize((248,248),Image.Resampling.BILINEAR))[12:236,12:236]
    if flip:m=m[:,::-1]
    # Mean over each 16x16 patch.
    weights=m.reshape(14,16,14,16).mean((1,3))
    assert weights.shape==(14,14) and np.isfinite(weights).all() and weights.min()>=0 and weights.max()<=1
    return np.ascontiguousarray(weights)

# Pools the last-block tokens z of model m, (batch, 1568, 384) with 1568 = 8 time steps x 14 x 14
# patches, into (batch, 5, 384): one vector per region of REGIONS, each passed through the
# encoder's final LayerNorm (fc_norm). Without weights, or when every weight is 1, this is P0.
def pool_tokens(m,z,weights=None):
    assert z.shape[1:]==(1568,384)
    # Token order is (time, patch row, patch column), as the patch embedding flattens it.
    grid=z.reshape(len(z),8,14,14,384)
    # P0: the mean over all tokens (the upstream model's own pooled feature), then each 7x7
    # quadrant averaged over time and patches.
    plain=[m.fc_norm(z.mean(1))]
    for h,w in [(0,0),(0,7),(7,0),(7,7)]:plain.append(m.fc_norm(grid[:,:,h:h+7,w:w+7].mean((1,2,3))))
    p0=torch.stack(plain,1)
    if weights is None or bool(torch.all(weights==1)):return p0
    # P1: weights are (batch, 14, 14) in [0, 1], one value per patch for all time steps.
    assert weights.shape==(len(z),14,14) and bool(torch.isfinite(weights).all())
    assert bool((weights>=0).all()) and bool((weights<=1).all())
    outputs=[]
    for i,(h0,h1,w0,w1) in enumerate(REGIONS):
        # The region's weights repeated over the 8 time steps, (batch, 8, rows, columns, 1).
        ww=weights[:,None,h0:h1,w0:w1,None].expand(-1,8,-1,-1,-1)
        zz=grid[:,:,h0:h1,w0:w1]
        # Weighted mean of the region's tokens, then the final LayerNorm.
        denominator=ww.sum((1,2,3));mean=(zz*ww).sum((1,2,3))/denominator.clamp_min(1e-12)
        weighted=m.fc_norm(mean)
        region=weights[:,h0:h1,w0:w1]
        # Per clip, the P0 value is used as is when the region has no pixel inside the frame
        # (all weights 0) or lies entirely inside it (all weights 1).
        fallback=(denominator[:,0]==0)|(region==1).all((1,2))
        outputs.append(torch.where(fallback[:,None],plain[i],weighted))
    out=torch.stack(outputs,1);assert bool(torch.isfinite(out).all());return out

# Encodes x, (batch, 3, 16, 224, 224) from fd15_vit.preprocess: prefix() applies the patch
# embedding and blocks 0-9, the loop the last two blocks. Returns (P0, P1), (batch, 5, 384)
# each; with weights=None or all ones, P1 equals P0.
def extract_pair(m,x,weights):
    z=prefix(m,x)
    for block in m.blocks[10:]:z=block(z)
    p0=pool_tokens(m,z)
    p1=pool_tokens(m,z,weights)
    return p0,p1
