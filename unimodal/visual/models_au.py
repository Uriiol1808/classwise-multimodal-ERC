import os
import re
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from dataclasses import dataclass
from typing import Optional, List

import torch
from facenet_pytorch import MTCNN

# ── IEMOCAP session constants ─────────────────────────────────────────────────
IEMOCAP_TRAIN_SESSIONS = [1, 2, 3, 4]
IEMOCAP_TEST_SESSIONS  = [5]
_TRANS_LINE_RE = re.compile(r"^(\S+)\s+\[\d+\.\d+-\d+\.\d+\]:\s*(.*)")
_EMO_LINE_RE   = re.compile(r"^\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+(\S+)\s+(\w+)")


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class AUConfig:
    device:    str = "cuda" if torch.cuda.is_available() else "cpu"
    au_model:  str = "svm"       # py-feat AU model: svm, xgb, rf, jaanet, drml
    num_frames: int = 16
    face_margin: float = 0.2
    use_fullframe_fallback: bool = False
    scene_cut_threshold: float = 0.3
    # ASD (Active Speaker Detection) — optional
    asd_weights:        Optional[str]   = None
    asd_threshold:      float           = 0.5
    asd_iou_threshold:  float           = 0.5
    asd_max_seconds:    Optional[float] = 60.0


# ── Utility functions (inlined — no dependency on utils.py) ──────────────────

def sample_frame_indices(n_total: int, n_frames: int) -> np.ndarray:
    """Evenly sample n_frames indices from [0, n_total)."""
    if n_total <= 0:
        return np.array([], dtype=int)
    if n_total <= n_frames:
        return np.arange(n_total)
    return np.linspace(0, n_total - 1, n_frames).astype(int)


def read_frame_at(cap: cv2.VideoCapture, fi: int) -> Optional[np.ndarray]:
    """Seek to frame fi and return RGB numpy array, or None on failure."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def pick_face_index(boxes: np.ndarray,
                    prev_box: Optional[np.ndarray],
                    iou_threshold: float = 0.30) -> int:
    """
    Select the best face box index from MTCNN detections.
    - First frame (prev_box=None): pick the largest face by area.
    - Subsequent frames: pick the box with highest IoU to the previous box;
      return -1 if the best IoU is below iou_threshold (scene cut / lost tracking).
    """
    if len(boxes) == 0:
        return -1

    if prev_box is None:
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        return int(np.argmax(areas))

    def _iou(a, b):
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0.0:
            return 0.0
        union = ((a[2]-a[0])*(a[3]-a[1]) +
                 (b[2]-b[0])*(b[3]-b[1]) - inter)
        return inter / (union + 1e-9)

    ious = [_iou(b, prev_box) for b in boxes]
    best = int(np.argmax(ious))
    return best if ious[best] >= iou_threshold else -1


def apply_margin_xyxy(box: np.ndarray, margin: float,
                      w: int, h: int) -> tuple:
    """Expand a face bounding box by a fractional margin, clamped to frame."""
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * margin))
    y1 = max(0, int(y1 - bh * margin))
    x2 = min(w, int(x2 + bw * margin))
    y2 = min(h, int(y2 + bh * margin))
    return x1, y1, x2, y2


def _frame_histogram(frame_rgb: np.ndarray, bins: int = 32) -> np.ndarray:
    hsv  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins],
                        [0, 180, 0, 256]).flatten().astype(np.float32)
    return hist / (hist.sum() + 1e-9)


def _is_scene_cut(hist_a: np.ndarray, hist_b: np.ndarray,
                  threshold: float = 0.3) -> bool:
    diff  = hist_a - hist_b
    chi2  = float(np.sum(diff ** 2 / (hist_a + hist_b + 1e-9)))
    return chi2 > threshold


# ── py-feat AU loader ─────────────────────────────────────────────────────────

def load_au_models(cfg: AUConfig):
    """
    Returns (mtcnn, feat_detector, embed_dim, asd).
    embed_dim = n_aus * 3  (mean + std + delta pooling).
    """
    mtcnn = MTCNN(keep_all=True, device=cfg.device)

    try:
        from feat import Detector
    except ImportError:
        raise ImportError(
            "py-feat is required.\n"
            "  pip install py-feat\n"
            "  # then patch scipy if needed:\n"
            "  sed -i 's/from scipy.integrate import simps/"
            "from scipy.integrate import simpson as simps/' "
            "$(find /path/to/env -name stats.py -path '*/feat/*')"
        )

    device_str = "cuda" if cfg.device.startswith("cuda") else "cpu"
    feat_detector = Detector(au_model=cfg.au_model,
                             emotion_model="svm",
                             device=device_str)
    print(f"  py-feat Detector loaded (au_model={cfg.au_model}, device={device_str})")

    n_aus    = _probe_n_aus(feat_detector)
    embed_dim = n_aus * 3
    print(f"  n_aus={n_aus}  embed_dim={embed_dim}")

    asd = None
    if cfg.asd_weights is not None:
        try:
            import sys as _sys
            for _p in ["/home/Imatge/oriol/unimodal/visual",
                       "/home/Imatge/oriol/unimodal/visual/Light-ASD"]:
                if _p not in _sys.path:
                    _sys.path.insert(0, _p)
            from visual.asd import LightASDWrapper
            asd = LightASDWrapper(weights_path=cfg.asd_weights, device=cfg.device)
            print(f"  Light-ASD loaded: {cfg.asd_weights}")
        except Exception as e:
            print(f"  WARNING: ASD disabled. ({e})")

    return mtcnn, feat_detector, embed_dim, asd


def _probe_n_aus(feat_detector) -> int:
    """Detect AU count from a blank image; fallback to 20."""
    try:
        dummy  = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        result = feat_detector.detect_image(dummy)
        if result is not None and len(result) > 0:
            au_cols = [c for c in result.columns if c.startswith("AU")]
            if au_cols:
                return len(au_cols)
    except Exception:
        pass
    return 20


# ── Per-frame AU extraction ───────────────────────────────────────────────────

def _extract_au_frame(
    crop_rgb: np.ndarray,
    feat_detector,
    full_frame_rgb: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Extract AU values from an MTCNN face crop.
    Falls back to full_frame_rgb if py-feat finds no face in the crop.
    Returns (n_aus,) float32, or None on complete failure.
    """
    sources = [crop_rgb]
    if full_frame_rgb is not None:
        sources.append(full_frame_rgb)

    for src in sources:
        try:
            result = feat_detector.detect_image(Image.fromarray(src))
            if result is None or len(result) == 0:
                continue
            au_cols = [c for c in result.columns if c.startswith("AU")]
            if not au_cols:
                continue
            vals = result[au_cols].values[0].astype(np.float32)
            if not np.isnan(vals).all():
                return np.nan_to_num(vals, nan=0.0)
        except Exception:
            continue
    return None


def _pool_aus(au_list: List[np.ndarray], embed_dim: int) -> np.ndarray:
    """
    mean + std + delta pooling of per-frame AU vectors → (embed_dim,).
    Same strategy as landmarks_exp: captures what AUs fired, how much
    they varied, and in which direction they changed across the utterance.
    """
    if not au_list:
        return np.zeros((embed_dim,), dtype=np.float32)
    stack = np.stack(au_list, axis=0)                    # (N, n_aus)
    mean  = stack.mean(axis=0)
    std   = stack.std(axis=0)
    delta = (np.diff(stack, axis=0).mean(axis=0)
             if len(stack) > 1
             else np.zeros(stack.shape[1], dtype=np.float32))
    return np.concatenate([mean, std, delta]).astype(np.float32)


# ── Main utterance-level extraction ──────────────────────────────────────────

def extract_utterance_au_embedding(
    video_path:             str,
    mtcnn:                  MTCNN,
    feat_detector,
    embed_dim:              int,
    num_frames:             int            = 16,
    face_margin:            float          = 0.2,
    use_fullframe_fallback: bool           = False,
    scene_cut_threshold:    float          = 0.3,
    asd                                    = None,
    asd_threshold:          float          = 0.5,
    asd_iou_threshold:      float          = 0.5,
    asd_max_seconds:        Optional[float]= 60.0,
    start_sec:              Optional[float]= None,
    end_sec:                Optional[float]= None,
) -> np.ndarray:
    """
    Extract Action Unit embedding for one utterance video (or time segment).

    Pipeline:
      1. Sample num_frames evenly from the utterance window.
      2. MTCNN detects faces; IoU heuristic tracks the speaker across frames.
      3. Each face crop → py-feat AU detector → (n_aus,) per frame.
      4. Pool: mean + std + delta → (embed_dim,).

    Returns zeros on any failure (missing video, no face detected, etc.).
    """
    zero = np.zeros((embed_dim,), dtype=np.float32)
    if not os.path.exists(video_path):
        return zero

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return zero

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if start_sec is not None or end_sec is not None:
        fs = int((start_sec or 0.0) * fps)
        fe = int((end_sec   or (total / fps)) * fps)
        fs = max(0, min(fs, total - 1))
        fe = max(fs + 1, min(fe, total))
        idxs = fs + sample_frame_indices(fe - fs, num_frames)
    else:
        idxs = sample_frame_indices(total, num_frames)

    # ── ASD path ──────────────────────────────────────────────────────────────
    if asd is not None:
        cap.release()
        speaker_boxes = asd.get_speaker_boxes(
            video_path=video_path, frame_indices=idxs, mtcnn=mtcnn,
            iou_threshold=asd_iou_threshold, asd_threshold=asd_threshold,
            max_seconds=(end_sec - start_sec + 2.0)
                        if (start_sec is not None and end_sec is not None)
                        else asd_max_seconds,
            start_sec=start_sec or 0.0,
        )
        return _aus_with_boxes(
            video_path=video_path, frame_indices=idxs,
            speaker_boxes=speaker_boxes, feat_detector=feat_detector,
            embed_dim=embed_dim, face_margin=face_margin,
            use_fullframe_fallback=use_fullframe_fallback,
        )

    # ── IoU heuristic path ────────────────────────────────────────────────────
    au_list:   List[np.ndarray]      = []
    prev_box:  Optional[np.ndarray]  = None
    prev_hist: Optional[np.ndarray]  = None

    for fi in idxs:
        frame_rgb = read_frame_at(cap, int(fi))
        if frame_rgb is None:
            continue

        curr_hist = _frame_histogram(frame_rgb)
        if prev_hist is not None and _is_scene_cut(
                prev_hist, curr_hist, scene_cut_threshold):
            prev_box = None
        prev_hist = curr_hist

        h, w      = frame_rgb.shape[:2]
        boxes, _  = mtcnn.detect(frame_rgb)
        fallback  = frame_rgb if use_fullframe_fallback else None

        if boxes is None or len(boxes) == 0:
            if use_fullframe_fallback:
                au = _extract_au_frame(frame_rgb, feat_detector)
                if au is not None:
                    au_list.append(au)
                prev_box = None
            continue

        sel = pick_face_index(boxes, prev_box)
        if sel < 0:
            if use_fullframe_fallback:
                au = _extract_au_frame(frame_rgb, feat_detector)
                if au is not None:
                    au_list.append(au)
                prev_box = None
            continue

        prev_box       = boxes[sel]
        x1, y1, x2, y2 = apply_margin_xyxy(prev_box, face_margin, w, h)
        crop            = frame_rgb[y1:y2, x1:x2]

        if crop.size == 0:
            if use_fullframe_fallback:
                au = _extract_au_frame(frame_rgb, feat_detector)
                if au is not None:
                    au_list.append(au)
                prev_box = None
            continue

        au = _extract_au_frame(crop, feat_detector, full_frame_rgb=fallback)
        if au is not None:
            au_list.append(au)

    cap.release()
    return _pool_aus(au_list, embed_dim)


def _aus_with_boxes(
    video_path:             str,
    frame_indices:          np.ndarray,
    speaker_boxes:          dict,
    feat_detector,
    embed_dim:              int,
    face_margin:            float,
    use_fullframe_fallback: bool,
) -> np.ndarray:
    """ASD path: extract AUs using pre-computed per-frame speaker boxes."""
    cap     = cv2.VideoCapture(video_path)
    au_list: List[np.ndarray] = []

    for fi in frame_indices:
        frame_rgb = read_frame_at(cap, int(fi))
        if frame_rgb is None:
            continue

        box  = speaker_boxes.get(int(fi))
        h, w = frame_rgb.shape[:2]

        if box is not None:
            x1, y1, x2, y2 = apply_margin_xyxy(box, face_margin, w, h)
            crop = frame_rgb[y1:y2, x1:x2]
            src  = crop if crop.size > 0 else (
                   frame_rgb if use_fullframe_fallback else None)
        else:
            src = frame_rgb if use_fullframe_fallback else None

        if src is None:
            continue

        fallback = (frame_rgb
                    if use_fullframe_fallback and src is not frame_rgb
                    else None)
        au = _extract_au_frame(src, feat_detector, full_frame_rgb=fallback)
        if au is not None:
            au_list.append(au)

    cap.release()
    return _pool_aus(au_list, embed_dim)


# ── IEMOCAP data loader ───────────────────────────────────────────────────────

import pandas as pd


def _parse_transcription(path: Path) -> list:
    entries = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            m = _TRANS_LINE_RE.match(line)
            if m:
                entries.append((m.group(1), m.group(2)))
    return entries


def _parse_emo_eval(path: Path) -> dict:
    times = {}
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = _EMO_LINE_RE.match(line.strip())
            if m:
                uid = m.group(3)
                if uid not in times:
                    times[uid] = (float(m.group(1)), float(m.group(2)))
    return times


def _speaker_from_utt_id(utt_id: str) -> str:
    part = utt_id.rsplit("_", 1)[-1]
    return part[0] if part and part[0] in ("M", "F") else "U"


def load_iemocap_au_dataframes(
    iemocap_root: str,
) -> tuple:
    """
    Parse IEMOCAP into (train_df, empty_dev_df, test_df, id_map).
    Columns: Dialogue_ID, Utterance_ID, Video_Path, Start_Time,
             End_Time, Speaker, Utt_Str_ID
    Sessions 1–4 → train, Session 5 → test.
    """
    root             = Path(iemocap_root)
    rows:     list   = []
    id_map:   dict   = {}
    dialogue_counter = 0

    for sessions in [IEMOCAP_TRAIN_SESSIONS, IEMOCAP_TEST_SESSIONS]:
        for sess in sessions:
            trans_dir    = root / f"Session{sess}" / "dialog" / "transcriptions"
            emo_eval_dir = root / f"Session{sess}" / "dialog" / "EmoEvaluation"
            avi_dir      = root / f"Session{sess}" / "dialog" / "avi" / "DivX"

            for trans_file in sorted(
                f for f in trans_dir.glob("*.txt")
                if not f.name.startswith("._")
            ):
                dialog_name = trans_file.stem
                emo_file    = emo_eval_dir / f"{dialog_name}.txt"
                timestamps  = _parse_emo_eval(emo_file) if emo_file.exists() else {}
                entries     = _parse_transcription(trans_file)
                if not entries:
                    continue

                entries.sort(
                    key=lambda x: timestamps.get(x[0], (float("inf"), 0))[0]
                )
                avi_path       = avi_dir / f"{dialog_name}.avi"
                id_map[dialogue_counter] = dialog_name

                for u_idx, (utt_str_id, _) in enumerate(entries):
                    start, end = timestamps.get(utt_str_id, (0.0, 0.0))
                    rows.append({
                        "Dialogue_ID":  dialogue_counter,
                        "Utterance_ID": u_idx,
                        "Video_Path":   str(avi_path),
                        "Start_Time":   start,
                        "End_Time":     end,
                        "Speaker":      _speaker_from_utt_id(utt_str_id),
                        "Utt_Str_ID":   utt_str_id,
                    })
                dialogue_counter += 1

    df_all     = pd.DataFrame(rows)
    train_mask = df_all["Utt_Str_ID"].str.startswith(
        tuple(f"Ses0{s}" for s in IEMOCAP_TRAIN_SESSIONS)
    )
    train_df     = df_all[train_mask].reset_index(drop=True)
    test_df      = df_all[~train_mask].reset_index(drop=True)
    empty_dev_df = pd.DataFrame(columns=df_all.columns)

    print(f"IEMOCAP: {len(train_df)} train / {len(test_df)} test utterances "
          f"across {dialogue_counter} dialogues.")
    return train_df, empty_dev_df, test_df, id_map