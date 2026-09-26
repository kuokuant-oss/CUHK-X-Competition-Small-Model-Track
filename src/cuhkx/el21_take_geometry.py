"""Isolated EL21 recording geometry diagnostic; historical takes.py stays immutable.

Recording membership uses only a verified 100 ms frame clock. No action, subject,
trial, clip ID, model probability, or leaderboard score is used to infer membership.
Reject inconsistent clocks; a delivery adapter must explicitly handle rejection.
"""
# Role: groups clips into takes by recorder start time. clock_origin checks that a clip's span
# follows an exact 100 ms frame clock and returns its take key (date, first timestamp - 100 ms *
# first frame counter); group_clock collects the clips of each key and orders them in time.
# Used by: el21_take_runtime.take_frame (clock_origin, and group_clock with order='start_end');
# inference. group_start_counter and partition are not used by the delivered run.
from collections import defaultdict


def clock_origin(span):
    # A span is (date, start seconds, first frame counter, end seconds, last frame counter) as
    # returned by takes.clip_span; seconds count from midnight.
    date, start, first, end, last = span
    # Integer milliseconds make the clock test below exact.
    start_ms, end_ms = round(start * 1000), round(end * 1000)
    if last < first or end_ms < start_ms:
        raise ValueError('Nonmonotone clip span')
    # At 10 fps the timestamp must advance by exactly 100 ms per counter step across the clip.
    if end_ms - start_ms != 100 * (last - first):
        raise ValueError('Span does not have an exact 100 ms frame clock')
    # The key is the date and the time (ms since midnight) at which the counter was 0, i.e. the
    # recorder start time; the clips of one take share it.
    return date, start_ms - 100 * first


def group_clock(spans, order='start_end'):
    # order sets the clip order inside a take: 'start_end' (the delivered run) sorts by (start,
    # end), 'stable_start' by start only, and 'reverse_identical' sorts like 'start_end' but puts
    # clips with identical (start, end) in reverse order (a diagnostic variant).
    if order not in ('stable_start', 'start_end', 'reverse_identical'):
        raise ValueError(order)
    # One group per take key. An invalid span raises here; take_frame passes only valid spans.
    grouped = defaultdict(list)
    for clip, span in spans.items():
        grouped[clock_origin(span)].append(clip)
    result = {}
    # Takes are emitted in (date, recorder start time) order.
    for key, members in sorted(grouped.items()):
        # spans[c][1] is the clip's start and spans[c][3] its end, in seconds. The sort is stable,
        # so clips with equal sort keys keep their order in spans.
        members.sort(key=lambda c: (spans[c][1],) if order == 'stable_start'
                     else (spans[c][1], spans[c][3]))
        if order == 'reverse_identical':
            # Bucket the sorted clips by (start, end), then reverse the clips within each bucket.
            tied = defaultdict(list)
            for c in members:
                tied[(spans[c][1], spans[c][3])].append(c)
            members = [c for group in tied.values() for c in reversed(group)]
        # The take id encodes the key: clock_<date>_<recorder start in ms since midnight>.
        result[f'clock_{key[0]}_{key[1]}'] = members
    return result


def group_start_counter(spans):
    # It is not used by the delivered run. This alternative sequential rule sorts clips by (date,
    # start) and opens a new take on a new date, on a first frame counter lower than the previous
    # clip's first counter, or when more than 60 s pass between the previous clip's end and this
    # clip's start. (takes.group_takes compares with the previous clip's last counter instead.)
    result, current, previous = {}, [], None
    for clip, span in sorted(spans.items(), key=lambda kv: (kv[1][0], kv[1][1])):
        new = (previous is None or span[0] != previous[0]
               or span[2] < previous[2] or span[1] - previous[3] > 60)
        if new and current:
            result[f'start_{len(result):04d}'] = current
            current = []
        current.append(clip)
        previous = span
    if current:
        result[f'start_{len(result):04d}'] = current
    return result


def partition(groups):
    # It is not used by the delivered run. It turns a grouping into a set of member sets, so that
    # two groupings can be compared regardless of take names and clip order.
    return frozenset(frozenset(m) for m in groups.values())
