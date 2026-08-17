import math
from typing import Dict, Tuple, List

## Helpers 

WIN_SEC = 1.2        # minimum continuous duration of the suspicious gesture in seconds
WRIST_HIP_MAX = 0.12 # Normalized wrist-to-hip distance (typical range is around 0.08-0.14)

def lm_dict(timestamp_obj) -> Dict[str, Tuple[float,float,float]]:
    landMark_dictionary = {}
    for lm in timestamp_obj.landmarks:
        landMark_dictionary[lm.name] = (lm.point.x, lm.point.y, lm.confidence)
    return landMark_dictionary

def ts_to_seconds(ts) -> float:
    sec = getattr(ts, "seconds", 0)
    nanos = getattr(ts, "nanos", 0)
    micros = getattr(ts, "microseconds", 0)
    return float(sec) + (float(nanos)/1e9) + (float(micros)/1e6)

def dist(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1]) # Distance between two (x, y) points

def frame_is_suspicious(lms: Dict[str, Tuple[float,float,float]]) -> bool:
    LW = lms.get("left_wrist");  RW = lms.get("right_wrist")
    LH = lms.get("left_hip");    RH = lms.get("right_hip")
    # suspicious if a wrist is close to its corresponding hip
    if LW and LH and dist((LW[0], LW[1]), (LH[0], LH[1])) <= WRIST_HIP_MAX:
        return True
    if RW and RH and dist((RW[0], RW[1]), (RH[0], RH[1])) <= WRIST_HIP_MAX:
        return True
    return False

def detect_suspicious_simple(track) -> List[Tuple[float,float]]:
    """Return [start, end] intervals (in seconds) where the wrist stays close to the hip for >= WIN_SEC."""
    if not track.timestamped_objects:
        return []
    frames = []
    for ts in track.timestamped_objects:
        t = ts_to_seconds(ts.time_offset)
        lms = lm_dict(ts)
        frames.append((t, frame_is_suspicious(lms)))

    intervals = []
    start = None
    for t, flag in frames:
        if flag and start is None:
            start = t
        if not flag and start is not None:
            if t - start >= WIN_SEC:
                intervals.append((start, t))
            start = None
    if start is not None and frames[-1][0] - start >= WIN_SEC:
        intervals.append((start, frames[-1][0]))
    return intervals

## Helpers
