"""Isolated clock-grouping delivery draft; no historical runtime replacement.

Invalid/missing clocks stay uncovered so the established submission decoder's
coverage threshold and per-clip raw fallback remain authoritative.
"""
# Role: builds the take table used at inference from clip spans: spans without an exact 100 ms
# frame clock are rejected, the rest are grouped by recorder start time, and a grouping report is
# attached to the table.
# Used by: el22_clock_adapter.from_raw, called by scripts/el25r_p1_repeat_runtime.py; inference.
import pandas as pd
from cuhkx.el21_take_geometry import clock_origin,group_clock

# One row per clip in a take: position is its 0-based index in take order and take_size the number
# of clips in the take; date, seconds (clip start, seconds since midnight) and frame (first frame
# counter) come from the clip's span.
TAKE_COLUMNS=['clip_id','take_id','position','take_size','date','seconds','frame']


def take_frame(spans):
    # spans maps clip id -> (date, start s, first counter, end s, last counter), from clip_span.
    # valid will map clip id -> span that passes the clock check; rejected, clip id -> reason.
    valid={};rejected={}
    # Step 1: check every span, in clip-id order. A rejected clip gets no row, so the decoder
    # treats it as outside every take and gives it its argmax.
    for clip in sorted(spans):
        span=spans[clip]
        # A failed clock check, a malformed span or a non-finite time rejects the clip.
        try:
            clock_origin(span)
        except (ValueError,TypeError,OverflowError) as exc:
            rejected[clip]=str(exc)
        else:
            valid[clip]=span
    # Step 2: group the valid spans by (date, recorder start time); each take is ordered by
    # (start, end).
    groups=group_clock(valid,'start_end')
    # Step 3: one row per clip in take order, or an empty table with the same columns.
    if groups:
        rows=[]
        for take,members in groups.items():
            for position,clip in enumerate(members):
                span=valid[clip]
                rows.append(dict(clip_id=clip,take_id=take,position=position,take_size=len(members),
                                 date=span[0],seconds=span[1],frame=span[2]))
        table=pd.DataFrame(rows,columns=TAKE_COLUMNS)
    else:
        table=pd.DataFrame(columns=TAKE_COLUMNS)
    # Step 4: the grouping report, which the entry point saves as take-clock-report.json: the rule
    # name, the numbers of input spans, accepted clips and takes, the rejected clips with reasons,
    # a description of the fallback, and identical_interval_ties, the number of accepted clips
    # that share (date, start, end) with an earlier accepted clip.
    table.attrs['clock_grouping']=dict(version='EL21-clock100ms-start-end-v1',input_spans=len(spans),
        accepted=len(valid),rejected=rejected,takes=len(groups),
        fallback='Uncovered clips use existing decoder raw fallback; below50% coverage all clips use raw argmax.',
        identical_interval_ties=sum(max(0,len(g)-1) for g in _ties(valid).values()))
    return table


def _ties(spans):
    # Clip ids grouped by (date, start seconds, end seconds); a group of two or more is a tie.
    result={}
    for c,s in spans.items():result.setdefault((s[0],s[1],s[3]),[]).append(c)
    return result
