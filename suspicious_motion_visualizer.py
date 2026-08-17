import os
from dotenv import load_dotenv
from google.cloud import videointelligence

import cv2
import math
import bisect
from typing import Dict, Tuple, List

# ==========================
# Parameters/Thresholds
# ==========================
WIN_SEC = 1.2        # minimum continuous duration of the suspicious gesture
WRIST_HIP_MAX = 0.12 # normalized wrist-to-hip distance threshold (adjust between 0.08 and 0.14)

# Simple skeleton connections (landmark_name, landmark_name)
POSE_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


# ==========================
# Landmark / time helpers
# ==========================
def _lm_dict(ts_obj) -> Dict[str, Tuple[float,float,float]]:
    d = {}
    for lm in ts_obj.landmarks:
        d[lm.name] = (lm.point.x, lm.point.y, lm.confidence)
    return d

def _ts_to_seconds(ts) -> float:
    # Video Intelligence may populate seconds/nanos or seconds/microseconds
    sec = getattr(ts, "seconds", 0)
    nanos = getattr(ts, "nanos", 0)
    micros = getattr(ts, "microseconds", 0)
    return float(sec) + (float(nanos)/1e9) + (float(micros)/1e6)

def _dist(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

def _frame_is_suspicious(lms: Dict[str, Tuple[float,float,float]]) -> bool:
    LW = lms.get("left_wrist");  RW = lms.get("right_wrist")
    LH = lms.get("left_hip");    RH = lms.get("right_hip")
    # suspicious if a wrist is close to its corresponding hip
    if LW and LH and _dist((LW[0], LW[1]), (LH[0], LH[1])) <= WRIST_HIP_MAX:
        return True
    if RW and RH and _dist((RW[0], RW[1]), (RH[0], RH[1])) <= WRIST_HIP_MAX:
        return True
    return False

def detect_suspicious_simple(track) -> List[Tuple[float,float]]:
    """Return [start, end] intervals (in seconds) where the wrist stays close to the hip for >= WIN_SEC."""
    if not track.timestamped_objects:
        return []
    frames = []
    for ts in track.timestamped_objects:
        t = _ts_to_seconds(ts.time_offset)
        lms = _lm_dict(ts)
        frames.append((t, _frame_is_suspicious(lms)))

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


# ==========================
# Cliente Video Intelligence (batch)
# ==========================
def configure_people_detection(
    include_bounding_boxes=True,
    include_attributes=True,
    include_pose_landmarks=True
):
    config = videointelligence.PersonDetectionConfig(
        include_bounding_boxes=include_bounding_boxes,
        include_attributes=include_attributes,
        include_pose_landmarks=include_pose_landmarks
    )
    return videointelligence.VideoContext(person_detection_config=config)

def load_video_content(file_path):
    with open(file_path, 'rb') as f:
        return f.read()

def run_videointel(video_path: str):
    load_dotenv()
    client = videointelligence.VideoIntelligenceServiceClient()
    ctx = configure_people_detection()

    op = client.annotate_video(
        request={
            "features": [videointelligence.Feature.PERSON_DETECTION],
            "input_content": load_video_content(video_path),
            "video_context": ctx,
        }
    )
    print("\nProcessing video (Person Detection + Pose Landmarks)...")
    res = op.result(timeout=600)
    print("Done.")
    return res.annotation_results[0]


# ==========================
# Preprocess tracks -> timeline per track
# ==========================
class TrackTimeline:
    """Structure for quickly looking up landmarks by time (seconds) in each track."""
    def __init__(self):
        self.times: List[float] = []
        self.landmarks: List[Dict[str, Tuple[float,float,float]]] = []
        self.bboxes: List[Tuple[float,float,float,float]] = []  # normalized (left, top, right, bottom)

    def add(self, t: float, lms: Dict[str, Tuple[float,float,float]], bbox_norm):
        self.times.append(t)
        self.landmarks.append(lms)
        if bbox_norm:
            self.bboxes.append((bbox_norm.left, bbox_norm.top, bbox_norm.right, bbox_norm.bottom))
        else:
            self.bboxes.append(None)

    def get_at(self, t: float):
        """Return the nearest index with time <= t (or None if not available)."""
        if not self.times:
            return None, None, None
        idx = bisect.bisect_right(self.times, t) - 1
        if idx < 0:
            return None, None, None
        return self.landmarks[idx], self.bboxes[idx], idx


# ==========================
# Drawing with OpenCV
# ==========================
def _to_px(xy_norm: Tuple[float,float], W: int, H: int) -> Tuple[int,int]:
    x = int(max(0, min(1, xy_norm[0])) * W)
    y = int(max(0, min(1, xy_norm[1])) * H)
    return x, y

def draw_landmarks(frame, lms: Dict[str, Tuple[float,float,float]]):
    H, W = frame.shape[:2]
    # points
    for name, (x, y, conf) in lms.items():
        if conf >= 0.3:
            px, py = _to_px((x, y), W, H)
            cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)

    # connections
    for a, b in POSE_CONNECTIONS:
        if a in lms and b in lms and lms[a][2] >= 0.3 and lms[b][2] >= 0.3:
            ax, ay = _to_px((lms[a][0], lms[a][1]), W, H)
            bx, by = _to_px((lms[b][0], lms[b][1]), W, H)
            cv2.line(frame, (ax, ay), (bx, by), (0, 200, 255), 2)

def draw_bbox(frame, bbox_norm):
    if not bbox_norm:
        return
    H, W = frame.shape[:2]
    x1, y1 = _to_px((bbox_norm[0], bbox_norm[1]), W, H)
    x2, y2 = _to_px((bbox_norm[2], bbox_norm[3]), W, H)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)

def is_time_in_intervals(t: float, intervals: List[Tuple[float,float]]) -> bool:
    for a, b in intervals:
        if a <= t <= b:
            return True
    return False


# ==========================
# Visualization: play the video and draw overlays
# ==========================
def play_with_overlays(video_path: str, annotation_result):
    # 1) Build timelines per track
    timelines: List[TrackTimeline] = []
    suspicious_map: List[List[Tuple[float,float]]] = []

    for person in annotation_result.person_detection_annotations:
        for track in person.tracks:
            tl = TrackTimeline()
            for ts in track.timestamped_objects:
                t = _ts_to_seconds(ts.time_offset)
                lms = _lm_dict(ts)
                bbox = getattr(ts, "normalized_bounding_box", None)
                tl.add(t, lms, bbox)
            timelines.append(tl)
            suspicious_map.append(detect_suspicious_simple(track))

    # 2) Play the video with OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open the video for playback/overlay.")
        return

    # fps and time step
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    win = "Landmarks (Video Intelligence) + Overlay"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t = frame_idx / max(1e-6, fps)  # seconds
            # If you want to synchronize with "segment.start_time_offset", add it here

            # For each track, find the nearest landmarks at t and draw them
            any_suspicious_now = False
            for tl, intervals in zip(timelines, suspicious_map):
                lms, bbox_norm, _ = tl.get_at(t)
                if lms:
                    draw_landmarks(frame, lms)
                    draw_bbox(frame, bbox_norm)
                if is_time_in_intervals(t, intervals):
                    any_suspicious_now = True

            # Status/time overlay
            txt = f"t={t:5.2f}s  FPS~{fps:.1f}"
            cv2.putText(frame, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240,240,240), 2, cv2.LINE_AA)

            if any_suspicious_now:
                cv2.putText(frame, "SUSPICIOUS HAND MOTION", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)
                # red border as an alert
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (2,2), (w-2,h-2), (0,0,255), 3)

            cv2.imshow(win, frame)
            frame_idx += 1

            # exit with 'q'
            if cv2.waitKey(int(1000 / max(1, fps))) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ==========================
# Main
# ==========================
def get_video_path():
    load_dotenv()
    video_path = os.getenv("VIDEO_PATH")
    if not video_path:
        raise ValueError("VIDEO_PATH is not set in .env")
    return video_path


def main():
    try:
        load_dotenv()
        video_path = get_video_path()
        annotation_result = run_videointel(video_path)
        play_with_overlays(video_path, annotation_result)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
