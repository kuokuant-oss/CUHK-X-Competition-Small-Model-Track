"""C0 clock grouping from raw filenames, with explicit metadata-only diagnostics."""
# Role: builds the test take table from the raw test folder: each clip's span is read from its
# frame file names (takes.clip_span) and the clips are grouped by recorder start time
# (el21_take_runtime.take_frame).
# Used by: scripts/el25r_p1_repeat_runtime.py, which calls from_raw with the default mode='normal';
# inference. The other modes are diagnostics that hide filename clocks to test the fallbacks; their
# results are recorded in evidence/runs/fallbacks/receipt.json.
from cuhkx.takes import clip_span
from cuhkx.el21_take_runtime import take_frame

# Each mode sets which file names takes.clip_span may read. clip_span reads Depth_Color names
# first, then IR names, then Skeleton/predictions JSON names; the diagnostic modes hide some of
# these sources:
#   normal: nothing is hidden (the delivered run);
#   Depth_hidden: Depth_Color is hidden, so the IR names are read instead;
#   IR_hidden: IR is hidden, which matters only for clips without Depth_Color stamps;
#   Skeleton_only: only the Skeleton names are visible;
#   partial_clock: every 7th clip in sorted order (index 0, 7, 14, ...) has no visible clock;
#   below50_clock: only every 3rd clip (index 0, 3, 6, ...) keeps its clock, which on the test
#     set puts take coverage (1/3) below the 50% threshold, so every clip gets its argmax;
#   all_clock_hidden: no clip has a visible clock, so the take table is empty.
# A clip whose clock is hidden falls outside every take. In every mode except normal, files lying
# directly in the clip folder (the last fallback of clip_span) are hidden too.
MODES=['normal','Depth_hidden','IR_hidden','Skeleton_only','partial_clock','below50_clock','all_clock_hidden']

# A stand-in for a clip folder: glob() answers only patterns whose first path component is an
# enabled subfolder, and finds nothing for any other pattern.
class Visible:
    def __init__(self,path,enabled):self.path,self.enabled=path,set(enabled)
    def glob(self,pattern):return self.path.glob(pattern) if pattern.split('/')[0] in self.enabled else []

def from_raw(root,mode='normal'):
    if mode not in MODES:raise ValueError(mode)
    # The SM_test_* clip folders under root, sorted by name; spans will map clip name -> span.
    clips=sorted(p for p in root.glob('SM_test_*') if p.is_dir());spans={}
    for i,clip in enumerate(clips):
        # Subfolders whose file names may be read; the diagnostic modes narrow this list.
        enabled=['Depth_Color','IR','Skeleton']
        if mode=='Depth_hidden':enabled=['IR','Skeleton']
        if mode=='IR_hidden':enabled=['Depth_Color','Skeleton']
        if mode=='Skeleton_only':enabled=['Skeleton']
        if mode=='partial_clock' and i%7==0:enabled=[]
        if mode=='below50_clock' and i%3!=0:enabled=[]
        if mode=='all_clock_hidden':enabled=[]
        # normal reads the real folder; the other modes read it through the Visible filter. A clip
        # without any readable stamp gets no span.
        span=clip_span(clip if mode=='normal' else Visible(clip,enabled))
        if span is not None:spans[clip.name]=span
    # Group the spans into takes; spans without an exact 100 ms clock are rejected there.
    table=take_frame(spans)
    # Add to the grouping report the names of all clip folders found, the mode and the name of the
    # grouping rule.
    table.attrs['clock_grouping'].update(visible_input_ids=[c.name for c in clips],metadata_visibility=mode,policy='clock100ms-start-end-v1',scored_ids_used=False)
    return table
