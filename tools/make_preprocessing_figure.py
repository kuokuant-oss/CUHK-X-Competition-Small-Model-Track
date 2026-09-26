"""Render the preprocessing figure of the technical report (Figure 2) for one training clip.

Every panel is computed with the package's own functions and the detector and palette table stored in
weights/model.pt, on the CPU; the crops are checked pixel for pixel against fused.load_clip_fused. The
optional --blur box is blurred in the IR frames before cropping, for publication only.

    python tools/make_preprocessing_figure.py --data /path/to/HAR/data \
        --clip 9_Pour_drinks/user17/5-2-1 --blur 250,160,345,245 --out outputs/preprocessing.pdf

Needs torch and torchvision (CPU builds suffice) and matplotlib, in addition to requirements.txt.
"""
# Role: renders the preprocessing figure of the technical report for one training clip on the
#   CPU, and asserts that its det and miw crops equal what the cache builder stores.
# Used by: run by hand for the report; not used by the delivered run.
import argparse
import base64
import hashlib
import sys
import tempfile
from pathlib import Path

# Package root; src/ and scripts/ go first on the import path, and no bytecode is written.
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

from cuhkx import depth as depth_codec, person  # noqa: E402
from cuhkx.budget import dequantize_state  # noqa: E402
from cuhkx.data import _decode_frames  # noqa: E402
from cuhkx.el22_support_pool import patch_support  # noqa: E402
from cuhkx.fd18_codec import load  # noqa: E402
from cuhkx.fused import crop_pad, load_clip_fused, square_window, tsn_picks  # noqa: E402
from cuhkx.images import center_crop  # noqa: E402


# (n, size, size, 4) uint8 frames: Depth_Color RGB and IR cut with the same window (edge pixels
# repeated outside the frame) and resized bilinearly, as in cuhkx.fused.load_clip_fused.
def crops(depth, ir, window, size):
    out = np.empty((len(depth), size, size, 4), np.uint8)
    for i in range(len(depth)):
        out[i, :, :, :3] = np.asarray(Image.fromarray(crop_pad(depth[i], *window)).resize((size, size), Image.BILINEAR))
        out[i, :, :, 3] = np.asarray(Image.fromarray(crop_pad(ir[i], *window)).resize((size, size), Image.BILINEAR))
    return out


# Gaussian blur of one (x0, y0, x1, y1) box of a frame, pasted back in place; for the published
# figure only.
def blur(frame, box):
    if box is None:
        return frame
    im = Image.fromarray(frame)
    region = im.crop(box).filter(ImageFilter.GaussianBlur(9))
    im.paste(region, box[:2], Image.new('L', region.size, 255).filter(ImageFilter.GaussianBlur(2)))
    return np.asarray(im)


def compute(data: Path, clip: str, pack: Path):
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large

    # Step 1: load the checkpoint (only its detector and metadata are used here) and the palette
    # table, checked by SHA256 and read through a temporary file.
    blob, _, _, _, det = load(pack)
    lut_bytes = base64.b64decode(blob['base']['meta']['depth_lut_base64'])
    assert hashlib.sha256(lut_bytes).hexdigest() == blob['base']['meta']['depth_lut_sha256']
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / 'lut.npz').write_bytes(lut_bytes)
        table = depth_codec.load_lut(Path(tmp) / 'lut.npz')

    # Step 2: decode every IR (grayscale) and Depth_Color (RGB) frame: n frames of h x w pixels.
    ir_dir, d_dir = data / 'IR' / clip, data / 'Depth_Color' / clip
    ir_files, d_files = sorted(ir_dir.glob('*.png')), sorted(d_dir.glob('*.png'))
    irs, _ = _decode_frames(ir_files, convert='L')
    deps, _ = _decode_frames(d_files, convert='RGB')
    ir, dep = np.stack(irs), np.stack(deps)
    n, (h, w) = len(ir), ir.shape[1:]
    assert n <= 32, 'the figure assumes a clip of at most 32 frames'

    # Step 3: person window from the checkpoint's detector on the probe frames, with the
    # full-frame pass only (the clip must be accepted there).
    model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)
    model.load_state_dict(dequantize_state(det))
    model.eval()
    probes = ir[person.probe_indices(n)]
    boxes, scores = person.detect_boxes(model, 'cpu', probes)
    kept, reason = person.accept_boxes(boxes, scores, w)
    assert len(kept), 'no detection; pick a clip that passes the first detector rung'
    window = person.window_from_boxes(kept, h, w)

    # Step 4: motion-in-window box: motion on decoded depth inside the in-frame part of the person
    # window, every 2nd pixel, top 2% of motion, 10% margin; mapped back to frame pixels and
    # squared.
    decoded = np.stack([depth_codec.decode(f, table)[0] for f in dep])
    valid = decoded != depth_codec.INVALID
    t, l, b, r = window
    ct, cl, cb, cr = max(0, t), max(0, l), min(h, b), min(w, r)
    sub_d, sub_v = decoded[:, ct:cb:2, cl:cr:2], valid[:, ct:cb:2, cl:cr:2]
    mt, ml, mb, mr = depth_codec.motion_bbox(sub_d, sub_v, quantile=0.98, margin=0.1, min_side=16)
    miw_window = square_window(ct + 2 * mt, cl + 2 * ml, ct + 2 * mb, cl + 2 * mr, 1.0)

    # Step 5: check the det and miw crops at 144 px.
    # The figure's crops must equal what the cache builder stores.
    ref_det, _ = load_clip_fused(d_dir, ir_dir, table, size=(144, 144), max_frames=32, window_override=window)
    ref_miw, stats = load_clip_fused(d_dir, ir_dir, table, size=(144, 144), max_frames=32, window_override=window,
                                     motion_margin=0.1, motion_in_window=True)
    # load_clip_fused must have used the motion box, not its fallback to the person window.
    assert stats['window_source'] == 'motion-in-override'
    assert np.array_equal(crops(dep, ir, window, 144), ref_det)
    assert np.array_equal(crops(dep, ir, miw_window, 144), ref_miw)

    # Step 6: motion map inside the window for panel (c) (mean absolute deviation from the
    # per-pixel temporal median, as in depth.motion_bbox) with its 98th-percentile threshold, and
    # the patch support weights of the transformer crop for panel (f).
    usable = sub_d.astype(np.float32)
    usable[~sub_v] = np.nan
    usable[:, ~sub_v.any(axis=0)] = 0.0
    motion = np.nan_to_num(np.nanmean(np.abs(usable - np.nanmedian(usable, axis=0)), axis=0), nan=0.0)
    return dict(ir=ir, depth=dep, decoded=decoded, valid=valid, window=window, miw_window=miw_window,
                boxes=boxes, motion=motion, motion_thr=np.quantile(motion[motion > 0], 0.98),
                motion_origin=(ct, cl), support=patch_support(h, w, window), n=n, reason=reason)


def plot(z, blur_box, out: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    # The IR frames (optionally blurred) and the inputs as the networks see them: det and miw
    # crops at 144 px, centre 128 px at the 32 member frame picks; det248 crops, centre 224 px at
    # the 16 transformer picks (tchw moves the channel axis for center_crop and back).
    ir = np.stack([blur(f, blur_box) for f in z['ir']])
    depth, n, win, mwin = z['depth'], z['n'], z['window'], z['miw_window']
    tchw = lambda a: np.ascontiguousarray(a.transpose(0, 3, 1, 2))
    p32, p16 = tsn_picks(n, 32, None), tsn_picks(n, 16, None)
    mdet = center_crop(tchw(crops(depth, ir, win, 144)[p32]), 128).transpose(0, 2, 3, 1)
    mmiw = center_crop(tchw(crops(depth, ir, mwin, 144)[p32]), 128).transpose(0, 2, 3, 1)
    vit = crops(depth, ir, win, 248)[p16][:, 12:236, 12:236]

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 6.8, 'axes.titlesize': 6.8,
                         'axes.linewidth': 0.4, 'pdf.fonttype': 42, 'xtick.labelsize': 5.8, 'ytick.labelsize': 6.2})
    # Layout in inches: axes() places an axis by its top-left corner, measured from the top-left
    # of the figure, and its size; note() writes text at such a point.
    FW, FH = 6.47, 5.28
    fig = plt.figure(figsize=(FW, FH))
    axes = lambda x, y, w, h: fig.add_axes([x / FW, 1 - (y + h) / FH, w / FW, h / FH])
    def note(x, y, text, **kw):
        fig.text(x / FW, 1 - y / FH, text, va='top', ha=kw.pop('ha', 'left'), fontsize=kw.pop('fs', 6.1),
                 color=kw.pop('color', '#333'), **kw)
    def clean(ax):
        ax.set_xticks([]); ax.set_yticks([])
    # Outline of a (top, left, bottom, right) window; a dashed one gets a dark underlay.
    def rect(ax, w, color, lw=1.1, ls='-'):
        t, l, b, r = w
        if ls == '--':
            ax.add_patch(Rectangle((l - .5, t - .5), r - l, b - t, fill=False, edgecolor='#222', lw=lw + .5, zorder=4, clip_on=False))
        ax.add_patch(Rectangle((l - .5, t - .5), r - l, b - t, fill=False, edgecolor=color, lw=lw, ls=ls, zorder=5, clip_on=False))
    # Panel with a full 640x480 frame and pad_rows of white space below it, so that windows
    # reaching past the bottom edge stay visible; returns the axis and its height.
    pad_rows = 70
    def raw(x, y, img, title, **kw):
        w = 2.09; h = w * (480 + pad_rows) / 640
        ax = axes(x, y, w, h); ax.imshow(img, interpolation='nearest', **kw)
        ax.set_xlim(-.5, 639.5); ax.set_ylim(479.5 + pad_rows, -.5); clean(ax)
        ax.add_patch(Rectangle((-.5, 479.5), 640, pad_rows, facecolor='white', edgecolor='none', zorder=1))
        ax.axhline(479.5, color='#999', lw=.4, zorder=2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, loc='left', pad=2.5)
        return ax, h

    # Row 1: (a) IR with the detector boxes and the person window, (b) Depth_Color with the same
    # window, (c) decoded depth with the top-2% motion pixels (red) and the motion-in-window box.
    y1 = 0.16
    ax, h1 = raw(0.0, y1, ir[0], '(a) IR 640×480, person detection', cmap='gray', vmin=0, vmax=255)
    for bx in z['boxes']:
        ax.add_patch(Rectangle((bx[0], bx[1]), bx[2] - bx[0], bx[3] - bx[1], fill=False, edgecolor='#39c0ff', lw=.45, zorder=3))
    rect(ax, win, 'white', 1.0, '--')
    ax, _ = raw(2.19, y1, depth[0], '(b) Depth_Color, same window')
    rect(ax, win, 'white', 1.0, '--')
    cm = matplotlib.colormaps['cividis'].copy(); cm.set_bad('white')
    ax, _ = raw(4.38, y1, np.ma.masked_invalid(np.where(z['valid'][0], z['decoded'][0].astype(float), np.nan)),
                '(c) decoded depth, motion in window', cmap=cm, vmin=0, vmax=254)
    mot, (ct, cl) = z['motion'], z['motion_origin']
    ax.imshow(np.ma.masked_where(mot < z['motion_thr'], np.ones_like(mot)), cmap=ListedColormap(['#ff2d2d']), vmin=0, vmax=1,
              zorder=3, interpolation='nearest', extent=(cl - .5, cl + 2 * mot.shape[1] - .5, ct + 2 * mot.shape[0] - .5, ct - .5))
    rect(ax, win, 'white', .8, '--'); rect(ax, mwin, '#ff9f1c', 1.2)

    # Row 2: the first input frame of (d) det and (e) miw at 128 px and (f) the transformer crop
    # with its 16-px patch grid, quadrant lines and patches shaded by their share outside the
    # camera frame; on det, a dotted line marks the frame's bottom edge when it is in the crop.
    y2 = y1 + h1 + 0.36
    sq, gp, gg = 1.02, 0.03, 0.12
    xs = [0, sq + gp, 2 * sq + gp + gg, 3 * sq + 2 * gp + gg, 4 * sq + 2 * gp + 2 * gg, 5 * sq + 3 * gp + 2 * gg]
    tiles = [(mdet[0][..., :3], None, 'Depth_Color', 'det'), (mdet[0][..., 3], 'gray', 'IR', 'det'),
             (mmiw[0][..., :3], None, 'Depth_Color', ''), (mmiw[0][..., 3], 'gray', 'IR', ''),
             (vit[0][..., :3], None, 'Depth_Color', 'vit'), (vit[0][..., 3], 'gray', 'IR ×3', 'vit')]
    edge_det = (480 - win[0]) / (win[2] - win[0]) * 144 - 8
    for x, (img, cmap, lab, kind) in zip(xs, tiles):
        ax = axes(x, y2, sq, sq); ax.imshow(img, cmap=cmap, vmin=0, vmax=255, interpolation='nearest'); clean(ax)
        size = img.shape[0]
        if kind == 'det' and edge_det < 128:
            ax.axhline(edge_det, color='white', lw=.7, ls=(0, (2, 1.5)))
        if kind == 'vit':
            for g in range(16, 224, 16):
                ax.axhline(g - .5, color='white', lw=.2, alpha=.55); ax.axvline(g - .5, color='white', lw=.2, alpha=.55)
            ax.axhline(111.5, color='#ff4d4f', lw=.8); ax.axvline(111.5, color='#ff4d4f', lw=.8)
            for (i, j), wgt in np.ndenumerate(z['support']):
                if wgt < 1:
                    ax.add_patch(Rectangle((j * 16 - .5, i * 16 - .5), 16, 16, facecolor='#ff4d4f', alpha=.6 * (1 - wgt), edgecolor='none'))
        ax.text(.03 * size, .97 * size, lab, fontsize=5.6, color='white', va='bottom', ha='left',
                bbox=dict(boxstyle='square,pad=0.15', fc='black', ec='none', alpha=.6))
    note(xs[0], y2 - .2, '(d) det view  4×128²', fs=6.8, color='black')
    note(xs[2], y2 - .2, '(e) motion-in-window view  4×128²', fs=6.8, color='black')
    note(xs[4], y2 - .2, '(f) ViT view  3×224², 14×14 patches', fs=6.8, color='black')

    # Row 3: (g) which native frames the 32 member picks (circles, one per use) and the 16
    # transformer picks (squares) take.
    y3 = y2 + sq + .40
    axt = axes(.62, y3, FW - .62, .36)
    axt.set_xlim(-.6, n - .4); axt.set_ylim(-.6, 1.6); axt.set_yticks([0, 1]); axt.set_yticklabels(['ViT: 16', 'CNN: 32'])
    axt.set_xticks(range(n)); axt.tick_params(length=1.5, pad=1)
    c32, c16 = np.bincount(p32, minlength=n), np.bincount(p16, minlength=n)
    for f in range(n):
        for k in range(c32[f]):
            axt.plot(f + (k - (c32[f] - 1) / 2) * .24, 1, 'o', ms=2.5, color='#1f77b4', mew=0)
        if c16[f]:
            axt.plot(f, 0, 's', ms=2.4, color='#2ca02c', mew=0)
    for side in ('top', 'right'):
        axt.spines[side].set_visible(False)
    axt.set_xlabel(f'native frame index ({n} frames ≈ {n / 10:.1f} s)', labelpad=1, fontsize=6.2)
    note(0, y3 - .2, f'(g) Temporal sampling of the {n} native frames: centre of each of 32 (CNN) or 16 (ViT) equal segments',
         fs=6.8, color='black')
    # Row 4: every 4th of the 32 det-view input frames (channels 0-2), labelled with the input
    # index and the native frame it came from.
    y4 = y3 + .36 + .33
    fw = (FW - .62 - 7 * .03) / 8
    for c, pos in enumerate(range(0, 32, 4)):
        ax = axes(.62 + c * (fw + .03), y4, fw, fw); ax.imshow(mdet[pos][..., :3], interpolation='nearest'); clean(ax)
        note(.62 + c * (fw + .03) + fw / 2, y4 + fw + .03, f'{pos} ← {int(p32[pos])}', fs=6.0, ha='center')
    note(0, y4 + .1, 'det-view\ninput,\nevery 4th\nof 32 frames\n(ch. 0–2)', fs=6.1)
    note(0, y4 + fw + .03, 'input ← frame', fs=6.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print('wrote', out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, required=True, help='training data root that holds IR/ and Depth_Color/')
    ap.add_argument('--clip', default='9_Pour_drinks/user17/5-2-1', help='action/user/trial')
    ap.add_argument('--pack', type=Path, default=ROOT / 'weights' / 'model.pt')
    ap.add_argument('--blur', default=None, help='x0,y0,x1,y1 box blurred in the IR frames, e.g. to hide a face')
    ap.add_argument('--out', type=Path, default=ROOT / 'outputs' / 'preprocessing.pdf')
    a = ap.parse_args()
    # Parse the optional --blur box, compute every panel, print a summary and draw the figure.
    box = tuple(int(v) for v in a.blur.split(',')) if a.blur else None
    z = compute(a.data, a.clip, a.pack)
    print(f"clip {a.clip}: {z['n']} frames, detection {z['reason']}, window {z['window']}, motion-in-window {z['miw_window']}")
    plot(z, box, a.out)


if __name__ == '__main__':
    main()
