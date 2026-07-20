import cv2
import math
import numpy as np
import subprocess
import tempfile
import os
from typing import Optional, Tuple, List


# ─── Video helpers ────────────────────────────────────────────────────────────

def sample_frame_indices(total_frames: int, num_samples: int) -> np.ndarray:
    if total_frames <= 0:
        return np.array([], dtype=int)
    if total_frames <= num_samples:
        return np.arange(total_frames, dtype=int)
    return np.linspace(0, total_frames - 1, num=num_samples, dtype=int)


def get_total_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def read_frame_at(cap: cv2.VideoCapture, frame_index: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame_bgr = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def read_all_frames(video_path: str,
                    max_seconds: Optional[float] = None,
                    start_sec: float = 0.0) -> Tuple[List[np.ndarray], float]:
    """
    Reads frames from a video up to max_seconds, then stops.
    Returns (frames_rgb, fps). Used by ASD which needs full temporal sequence.

    start_sec: seek to this position before reading (IEMOCAP: utterance start).
               Default 0.0 → MELD behaviour unchanged.
    max_seconds must be applied HERE (early exit during reading) not after
    loading all frames — a 5-minute 1080p video is ~44GB if fully loaded.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if start_sec > 0.0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
    max_frames = int(max_seconds * fps) if max_seconds and max_seconds > 0 else None
    frames = []
    while True:
        if max_frames is not None and len(frames) >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


# ─── Audio helpers ────────────────────────────────────────────────────────────

def extract_audio_mfcc(video_path: str, sr: int = 16000, n_mfcc: int = 13,
    hop_length: int = 160,   # 10ms at 16kHz
    win_length: int = 400,   # 25ms at 16kHz
) -> Optional[np.ndarray]:
    """
    Extracts MFCC features from the audio track of a video file.
    Returns (T, n_mfcc) float32 array, or None if extraction fails.

    Requires: ffmpeg in PATH, librosa.
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa is required for audio extraction: pip install librosa")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-ac", "1", "-ar", str(sr),
                "-vn", tmp_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0 or not os.path.exists(tmp_path):
            return None

        y, _ = librosa.load(tmp_path, sr=sr, mono=True)
        if len(y) == 0:
            return None

        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=n_mfcc,
            hop_length=hop_length, win_length=win_length,
        )  # (n_mfcc, T)
        return mfcc.T.astype(np.float32)  # (T, n_mfcc)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def extract_raw_audio(video_path: str, sr: int = 16000) -> Optional[np.ndarray]:
    """
    Extracts raw waveform from video. Returns (N,) float32 or None.
    Used by Light-ASD which expects raw audio aligned with video frames.
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa is required: pip install librosa")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-ac", "1", "-ar", str(sr),
                "-vn", tmp_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0 or not os.path.exists(tmp_path):
            return None

        y, _ = librosa.load(tmp_path, sr=sr, mono=True)
        return y.astype(np.float32) if len(y) > 0 else None

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ─── Face helpers ─────────────────────────────────────────────────────────────

def iou_xyxy(a, b) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-9))


def pick_face_index(boxes: np.ndarray, prev_box: Optional[np.ndarray]) -> int:
    """
    Fallback face selector (used when ASD is disabled):
      - If prev_box exists: pick max IoU with previous (temporal consistency)
      - Else: pick largest area
    """
    if boxes is None or len(boxes) == 0:
        return -1
    if prev_box is not None:
        ious = [iou_xyxy(boxes[i], prev_box) for i in range(len(boxes))]
        return int(np.argmax(ious))
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return int(np.argmax(areas))


def apply_margin_xyxy(
    box: np.ndarray, margin: float, w: int, h: int
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box.tolist()
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    mx = margin * bw
    my = margin * bh
    x1 = int(max(0, math.floor(x1 - mx)))
    y1 = int(max(0, math.floor(y1 - my)))
    x2 = int(min(w, math.ceil(x2 + mx)))
    y2 = int(min(h, math.ceil(y2 + my)))
    return x1, y1, x2, y2


def box_center(box: np.ndarray) -> Tuple[float, float]:
    return float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)