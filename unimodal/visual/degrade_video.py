"""
degrade_video.py

Applies frame-level degradation BEFORE ViT/landmark embedding for the
test set only. Supports IEMOCAP and MELD, and all three visual modes
(vit, landmarks, landmarks_exp).

Degradations (teleassistance setting):
  blur      — Gaussian blur (σ px), applied to the WHOLE frame before face
               detection. This is scene-level by design: motion blur,
               out-of-focus webcam, and H.264 compression affect the entire
               frame, not just the face region. At high σ, MTCNN may fail
               to detect the face → fullframe_fallback fires naturally.
  occlusion — Black rectangle over the lower part of the FACE BOUNDING BOX.
               FACE-AWARE: MTCNN (or LightASD for multi-speaker scenes)
               detects the speaker's face first; the lower occlude_frac of
               that detected box is then zeroed before the crop is passed
               to the encoder. This matches real lower-face occlusion
               (mask, hand) and is consistent with the training degradation
               pipeline (create_degraded_train_visual.py).

Usage
-----
# IEMOCAP — blur σ=3 with ViT (scene-level, unchanged)
python -m visual.degrade_video --dataset iemocap \
    --iemocap_root /media/ssd2/oriol/IEMOCAP/IEMOCAP_full_release \
    --visual_mode vit \
    --out /media/ssd2/oriol/IEMOCAP/.../degraded/iemocap_vit_blur_s3.pkl \
    --degradation blur --blur_sigma 3

# IEMOCAP — face-aware occlusion 50% with ViT
python -m visual.degrade_video --dataset iemocap \
    --iemocap_root /media/ssd2/oriol/IEMOCAP/IEMOCAP_full_release \
    --visual_mode vit \
    --out /media/ssd2/oriol/IEMOCAP/.../degraded/iemocap_vit_occlude50.pkl \
    --degradation occlusion --occlude_frac 0.5

# MELD — face-aware occlusion 50% with landmarks (+ optional LightASD)
python -m visual.degrade_video --dataset meld \
    --train_csv /media/ssd2/oriol/MELD/train_sent_emo.csv \
    --dev_csv   /media/ssd2/oriol/MELD/dev_sent_emo.csv \
    --test_csv  /media/ssd2/oriol/MELD/test_sent_emo.csv \
    --test_video_dir /media/ssd2/oriol/MELD/test \
    --visual_mode landmarks \
    --asd_weights /home/Imatge/oriol/unimodal/visual/pretrain_AVA_CVPR.model \
    --out /media/ssd2/oriol/MELD/embeddings/degraded/meld_lm_occlude50.pkl \
    --degradation occlusion --occlude_frac 0.5

Suggested sweeps (run once per visual_mode × dataset)
------------------------------------------------------
  blur      : --blur_sigma    1 3 7 15
  occlusion : --occlude_frac  0.25 0.5 0.75 1.0
"""

import os
import re
import io
import argparse
import pickle
import tempfile

import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm

from .models_landmarks_new import (
    VisualConfig,
    load_visual_models,
    load_iemocap_visual_dataframes,
    extract_utterance_visual_embedding,
    extract_utterance_landmark_embedding,
    _extract_landmark_frame,
    _pool_landmarks,
    _extract_exp_frame,
    _pool_exp,
)
from .models_au import (
    AUConfig,
    load_au_models,
    extract_utterance_au_embedding,
    _extract_au_frame,
    _pool_aus,
    _frame_histogram,
    _is_scene_cut,
)


# ── MELD CSV mojibake fix ──────────────────────────────────────────────────────

def _fix_meld_mojibake(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    def _replace_c1(m: re.Match) -> str:
        byte_val = ord(m.group(0))
        try:
            return bytes([byte_val]).decode("windows-1252")
        except (ValueError, UnicodeDecodeError):
            return m.group(0)
    fixed = re.sub(r"[\u0080-\u009f]", _replace_c1, text)
    return fixed.replace("\xa0", " ")


def load_csv(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        raw = f.read()
    return pd.read_csv(io.StringIO(_fix_meld_mojibake(raw)))


# ── Blur transform (scene-level — applied to whole frame, unchanged) ──────────

def make_blur_transform(blur_sigma: float):
    """
    Gaussian blur at sigma pixels, applied to the WHOLE frame.
    σ=1: subtle; σ=3: clear quality loss; σ=7: heavy; σ=15: near-unrecognisable.
    This is intentionally scene-level, not face-aware: motion blur and
    compression artefacts affect the entire frame, not just the face.
    """
    def transform(frame_rgb: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(frame_rgb, (0, 0), sigmaX=blur_sigma)
    return transform


# ── Face-aware occlusion helpers (mirrors create_degraded_train_visual.py) ───

def _sample_idxs(total, n_frames, start_sec=None, end_sec=None, fps=25.0):
    if start_sec is not None:
        fs = max(0, min(int(start_sec * fps), total - 1))
        fe = max(fs + 1, min(int((end_sec or total / fps) * fps), total))
    else:
        fs, fe = 0, total
    return np.linspace(fs, fe - 1, n_frames).astype(int)


def _read_frame(cap, fi):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
    ok, f = cap.read()
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if ok and f is not None else None


def _apply_margin(box, margin, w, h):
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    bw, bh = x2 - x1, y2 - y1
    return (max(0, int(x1 - bw * margin)), max(0, int(y1 - bh * margin)),
            min(w, int(x2 + bw * margin)), min(h, int(y2 + bh * margin)))


def _pick_face(boxes, prev_box, iou_thr=0.30):
    if len(boxes) == 0:
        return -1
    if prev_box is None:
        return int(np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]))
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-9)
    ious = [_iou(b, prev_box) for b in boxes]
    best = int(np.argmax(ious))
    return best if ious[best] >= iou_thr else -1


def _occlude_box(frame_rgb, x1, y1, x2, y2, occlude_frac):
    """
    Zero the lower `occlude_frac` of the (already margin-expanded) face box.
    Called AFTER _apply_margin, BEFORE cropping the face.
    """
    fh    = y2 - y1
    occ_h = max(1, int(fh * occlude_frac))
    out   = frame_rgb.copy()
    out[y2 - occ_h : y2, x1 : x2] = 0
    return out


# ── Face-aware ViT extraction (occlusion path) ────────────────────────────────

def _extract_vit_occlusion(
    video_path, mtcnn, processor, vit_model, device, embed_dim,
    num_frames, face_margin, use_fullframe_fallback,
    start_sec, end_sec, occlude_frac,
    asd=None, asd_threshold=0.5, asd_iou_threshold=0.5, asd_max_seconds=60.0,
):
    import torch
    zero = np.zeros(embed_dim, dtype=np.float32)
    if not os.path.exists(video_path):
        return zero

    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    idxs  = _sample_idxs(total, num_frames, start_sec, end_sec, fps)

    @torch.no_grad()
    def _embed(crop_rgb):
        inp = processor(images=Image.fromarray(crop_rgb), return_tensors="pt").to(device)
        out = vit_model(**inp, output_hidden_states=True)
        return out.hidden_states[-1][:, 0, :].squeeze(0).cpu().numpy()

    # ── ASD path (multi-speaker scenes) ───────────────────────────────────────
    if asd is not None:
        cap.release()
        t_window = ((end_sec - start_sec + 2.0)
                    if start_sec is not None and end_sec is not None
                    else asd_max_seconds)
        speaker_boxes = asd.get_speaker_boxes(
            video_path=video_path, frame_indices=idxs, mtcnn=mtcnn,
            iou_threshold=asd_iou_threshold, asd_threshold=asd_threshold,
            max_seconds=t_window, start_sec=start_sec or 0.0)
        cap2 = cv2.VideoCapture(video_path)
        embs = []
        for fi in idxs:
            frame = _read_frame(cap2, fi)
            if frame is None:
                continue
            box = speaker_boxes.get(int(fi))
            h, w = frame.shape[:2]
            if box is not None:
                x1, y1, x2, y2 = _apply_margin(box, face_margin, w, h)
                frame = _occlude_box(frame, x1, y1, x2, y2, occlude_frac)
                crop = frame[y1:y2, x1:x2]
                src  = crop if crop.size > 0 else (frame if use_fullframe_fallback else None)
            else:
                src = frame if use_fullframe_fallback else None
            if src is not None:
                embs.append(_embed(src))
        cap2.release()
        return np.mean(np.stack(embs), 0).astype(np.float32) if embs else zero

    # ── IoU heuristic path ────────────────────────────────────────────────────
    embs, prev_box = [], None
    for fi in idxs:
        frame = _read_frame(cap, fi)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        boxes, _ = mtcnn.detect(frame)
        if boxes is None or len(boxes) == 0:
            if use_fullframe_fallback:
                embs.append(_embed(frame))
            prev_box = None
            continue
        sel = _pick_face(boxes, prev_box)
        if sel < 0:
            if use_fullframe_fallback:
                embs.append(_embed(frame))
            prev_box = None
            continue
        prev_box = boxes[sel]
        x1, y1, x2, y2 = _apply_margin(prev_box, face_margin, w, h)
        frame = _occlude_box(frame, x1, y1, x2, y2, occlude_frac)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            if use_fullframe_fallback:
                embs.append(_embed(frame))
            continue
        embs.append(_embed(crop))
    cap.release()
    return np.mean(np.stack(embs), 0).astype(np.float32) if embs else zero


# ── Face-aware landmarks extraction (occlusion path) ──────────────────────────

def _extract_landmarks_occlusion(
    video_path, mtcnn, tddfa, embed_dim, visual_mode,
    num_frames, face_margin, use_fullframe_fallback,
    start_sec, end_sec, occlude_frac,
    asd=None, asd_threshold=0.5, asd_iou_threshold=0.5, asd_max_seconds=60.0,
):
    frame_fn = _extract_exp_frame if visual_mode == "landmarks_exp" else _extract_landmark_frame
    pool_fn  = _pool_exp          if visual_mode == "landmarks_exp" else _pool_landmarks
    zero = np.zeros(embed_dim, dtype=np.float32)
    if not os.path.exists(video_path):
        return zero

    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    idxs  = _sample_idxs(total, num_frames, start_sec, end_sec, fps)

    # ── ASD path ──────────────────────────────────────────────────────────────
    if asd is not None:
        cap.release()
        t_window = ((end_sec - start_sec + 2.0)
                    if start_sec is not None and end_sec is not None
                    else asd_max_seconds)
        speaker_boxes = asd.get_speaker_boxes(
            video_path=video_path, frame_indices=idxs, mtcnn=mtcnn,
            iou_threshold=asd_iou_threshold, asd_threshold=asd_threshold,
            max_seconds=t_window, start_sec=start_sec or 0.0)
        cap2 = cv2.VideoCapture(video_path)
        feats = []
        for fi in idxs:
            frame = _read_frame(cap2, fi)
            if frame is None:
                continue
            box = speaker_boxes.get(int(fi))
            h, w = frame.shape[:2]
            if box is not None:
                x1, y1, x2, y2 = _apply_margin(box, face_margin, w, h)
                frame = _occlude_box(frame, x1, y1, x2, y2, occlude_frac)
                crop = frame[y1:y2, x1:x2]
                src  = crop if crop.size > 0 else (frame if use_fullframe_fallback else None)
            else:
                src = frame if use_fullframe_fallback else None
            if src is not None:
                f = frame_fn(src, tddfa)
                if f is not None:
                    feats.append(f)
        cap2.release()
        return pool_fn(feats, embed_dim)

    # ── IoU heuristic path ────────────────────────────────────────────────────
    feats, prev_box = [], None
    for fi in idxs:
        frame = _read_frame(cap, fi)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        boxes, _ = mtcnn.detect(frame)
        if boxes is None or len(boxes) == 0:
            if use_fullframe_fallback:
                f = frame_fn(frame, tddfa)
                if f is not None:
                    feats.append(f)
            prev_box = None
            continue
        sel = _pick_face(boxes, prev_box)
        if sel < 0:
            if use_fullframe_fallback:
                f = frame_fn(frame, tddfa)
                if f is not None:
                    feats.append(f)
            prev_box = None
            continue
        prev_box = boxes[sel]
        x1, y1, x2, y2 = _apply_margin(prev_box, face_margin, w, h)
        frame = _occlude_box(frame, x1, y1, x2, y2, occlude_frac)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        f = frame_fn(crop, tddfa)
        if f is not None:
            feats.append(f)
    cap.release()
    return pool_fn(feats, embed_dim)


# ── Face-aware AU extraction (occlusion path) ──────────────────────────────────

def _extract_au_occlusion(
    video_path, mtcnn, feat_detector, embed_dim,
    num_frames, face_margin, use_fullframe_fallback,
    start_sec, end_sec, occlude_frac,
    asd=None, asd_threshold=0.5, asd_iou_threshold=0.5, asd_max_seconds=60.0,
):
    zero = np.zeros(embed_dim, dtype=np.float32)
    if not os.path.exists(video_path):
        return zero

    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    idxs  = _sample_idxs(total, num_frames, start_sec, end_sec, fps)

    # ── ASD path (multi-speaker scenes — important for MELD) ──────────────────
    if asd is not None:
        cap.release()
        t_window = ((end_sec - start_sec + 2.0)
                    if start_sec is not None and end_sec is not None
                    else asd_max_seconds)
        speaker_boxes = asd.get_speaker_boxes(
            video_path=video_path, frame_indices=idxs, mtcnn=mtcnn,
            iou_threshold=asd_iou_threshold, asd_threshold=asd_threshold,
            max_seconds=t_window, start_sec=start_sec or 0.0)
        cap2 = cv2.VideoCapture(video_path)
        au_list = []
        for fi in idxs:
            frame = _read_frame(cap2, fi)
            if frame is None:
                continue
            box = speaker_boxes.get(int(fi))
            h, w = frame.shape[:2]
            if box is not None:
                x1, y1, x2, y2 = _apply_margin(box, face_margin, w, h)
                frame = _occlude_box(frame, x1, y1, x2, y2, occlude_frac)
                crop = frame[y1:y2, x1:x2]
                src  = crop if crop.size > 0 else (frame if use_fullframe_fallback else None)
            else:
                src = frame if use_fullframe_fallback else None
            if src is not None:
                au = _extract_au_frame(src, feat_detector)
                if au is not None:
                    au_list.append(au)
        cap2.release()
        return _pool_aus(au_list, embed_dim)

    # ── IoU heuristic path ────────────────────────────────────────────────────
    au_list, prev_box, prev_hist = [], None, None
    for fi in idxs:
        frame = _read_frame(cap, fi)
        if frame is None:
            continue
        curr_hist = _frame_histogram(frame)
        if prev_hist is not None and _is_scene_cut(prev_hist, curr_hist):
            prev_box = None
        prev_hist = curr_hist
        h, w = frame.shape[:2]
        boxes, _ = mtcnn.detect(frame)
        if boxes is None or len(boxes) == 0:
            if use_fullframe_fallback:
                au = _extract_au_frame(frame, feat_detector)
                if au is not None:
                    au_list.append(au)
            prev_box = None
            continue
        sel = _pick_face(boxes, prev_box)
        if sel < 0:
            if use_fullframe_fallback:
                au = _extract_au_frame(frame, feat_detector)
                if au is not None:
                    au_list.append(au)
            prev_box = None
            continue
        prev_box = boxes[sel]
        x1, y1, x2, y2 = _apply_margin(prev_box, face_margin, w, h)
        frame = _occlude_box(frame, x1, y1, x2, y2, occlude_frac)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        au = _extract_au_frame(crop, feat_detector)
        if au is not None:
            au_list.append(au)
    cap.release()
    return _pool_aus(au_list, embed_dim)


# ── Blur path: temp degraded video approach (unchanged, scene-level) ─────────

def _write_degraded_segment(video_path, start_sec, end_sec, frame_transform):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if start_sec is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)

    tmp = tempfile.NamedTemporaryFile(suffix='.avi', delete=False)
    tmp_path = tmp.name
    tmp.close()

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
    end_ms = (end_sec * 1000.0) if end_sec is not None else float('inf')

    while True:
        if cap.get(cv2.CAP_PROP_POS_MSEC) > end_ms:
            break
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        degraded_rgb = frame_transform(frame_rgb)
        degraded_bgr = cv2.cvtColor(degraded_rgb, cv2.COLOR_RGB2BGR)
        writer.write(degraded_bgr)

    cap.release()
    writer.release()
    return tmp_path


def _extract_blur(video_path, mtcnn, processor, model, cfg, embed_dim,
                  asd, start_sec, end_sec, frame_transform):
    tmp_path = _write_degraded_segment(video_path, start_sec, end_sec, frame_transform)
    try:
        if cfg.visual_mode == "au":
            return extract_utterance_au_embedding(
                video_path=tmp_path, mtcnn=mtcnn,
                feat_detector=model, embed_dim=embed_dim,
                num_frames=cfg.num_frames, face_margin=cfg.face_margin,
                use_fullframe_fallback=cfg.use_fullframe_fallback,
                scene_cut_threshold=cfg.scene_cut_threshold,
                asd=asd, asd_threshold=cfg.asd_threshold,
                asd_iou_threshold=cfg.asd_iou_threshold,
                asd_max_seconds=cfg.asd_max_seconds,
                start_sec=None, end_sec=None,
            )
        elif cfg.visual_mode in ("landmarks", "landmarks_exp"):
            return extract_utterance_landmark_embedding(
                video_path=tmp_path, mtcnn=mtcnn, tddfa=model,
                embed_dim=embed_dim, num_frames=cfg.num_frames,
                face_margin=cfg.face_margin,
                use_fullframe_fallback=cfg.use_fullframe_fallback,
                scene_cut_threshold=cfg.scene_cut_threshold,
                asd=asd, asd_threshold=cfg.asd_threshold,
                asd_iou_threshold=cfg.asd_iou_threshold,
                asd_max_seconds=cfg.asd_max_seconds,
                start_sec=None, end_sec=None,
                visual_mode=cfg.visual_mode,
            )
        else:  # vit
            return extract_utterance_visual_embedding(
                video_path=tmp_path, mtcnn=mtcnn,
                processor=processor, model=model, device=cfg.device,
                embed_dim=embed_dim, num_frames=cfg.num_frames,
                face_margin=cfg.face_margin,
                use_fullframe_fallback=cfg.use_fullframe_fallback,
                scene_cut_threshold=cfg.scene_cut_threshold,
                asd=asd, asd_threshold=cfg.asd_threshold,
                asd_iou_threshold=cfg.asd_iou_threshold,
                asd_max_seconds=cfg.asd_max_seconds,
                start_sec=None, end_sec=None,
            )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _extract_occlusion(video_path, mtcnn, processor, model, cfg, embed_dim,
                       asd, start_sec, end_sec, occlude_frac):
    if cfg.visual_mode == "au":
        return _extract_au_occlusion(
            video_path=video_path, mtcnn=mtcnn, feat_detector=model,
            embed_dim=embed_dim,
            num_frames=cfg.num_frames, face_margin=cfg.face_margin,
            use_fullframe_fallback=cfg.use_fullframe_fallback,
            start_sec=start_sec, end_sec=end_sec, occlude_frac=occlude_frac,
            asd=asd, asd_threshold=cfg.asd_threshold,
            asd_iou_threshold=cfg.asd_iou_threshold,
            asd_max_seconds=cfg.asd_max_seconds,
        )
    elif cfg.visual_mode in ("landmarks", "landmarks_exp"):
        return _extract_landmarks_occlusion(
            video_path=video_path, mtcnn=mtcnn, tddfa=model,
            embed_dim=embed_dim, visual_mode=cfg.visual_mode,
            num_frames=cfg.num_frames, face_margin=cfg.face_margin,
            use_fullframe_fallback=cfg.use_fullframe_fallback,
            start_sec=start_sec, end_sec=end_sec, occlude_frac=occlude_frac,
            asd=asd, asd_threshold=cfg.asd_threshold,
            asd_iou_threshold=cfg.asd_iou_threshold,
            asd_max_seconds=cfg.asd_max_seconds,
        )
    else:  # vit
        return _extract_vit_occlusion(
            video_path=video_path, mtcnn=mtcnn, processor=processor,
            vit_model=model, device=cfg.device, embed_dim=embed_dim,
            num_frames=cfg.num_frames, face_margin=cfg.face_margin,
            use_fullframe_fallback=cfg.use_fullframe_fallback,
            start_sec=start_sec, end_sec=end_sec, occlude_frac=occlude_frac,
            asd=asd, asd_threshold=cfg.asd_threshold,
            asd_iou_threshold=cfg.asd_iou_threshold,
            asd_max_seconds=cfg.asd_max_seconds,
        )


# ── MELD test dataframe loader (unchanged) ────────────────────────────────────

def _load_meld_test_visual(train_csv, dev_csv, test_csv, test_video_dir):
    train_df = load_csv(train_csv)
    dev_df   = load_csv(dev_csv)
    test_df  = load_csv(test_csv)

    train_ids = sorted(train_df["Dialogue_ID"].astype(int).unique())
    dev_ids   = sorted(dev_df["Dialogue_ID"].astype(int).unique())
    test_ids  = sorted(test_df["Dialogue_ID"].astype(int).unique())

    offset_dev  = (max(train_ids) + 1) if train_ids else 0
    dev_map     = {d: offset_dev + i for i, d in enumerate(dev_ids)}
    offset_test = (max(dev_map.values()) + 1) if dev_map else offset_dev
    test_map    = {d: offset_test + i for i, d in enumerate(test_ids)}

    test_df = test_df.copy()
    test_df["Dialogue_ID"]  = test_df["Dialogue_ID"].astype(int)
    test_df["Utterance_ID"] = test_df["Utterance_ID"].astype(int)

    if "Video_Path" not in test_df.columns:
        test_df["Video_Path"] = test_df.apply(
            lambda r: os.path.join(
                test_video_dir,
                f"dia{int(r['Dialogue_ID'])}_utt{int(r['Utterance_ID'])}.mp4"
            ), axis=1
        )

    def _to_seconds(val) -> float:
        if pd.isna(val):
            return 0.0
        try:
            return float(val) / 1000.0
        except (ValueError, TypeError):
            pass
        s = str(val).replace(',', '.')
        parts = s.split(':')
        try:
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            return float(s)
        except ValueError:
            return 0.0

    for col_raw, col_s in [("StartTime", "Start_Time"), ("EndTime", "End_Time")]:
        if col_raw in test_df.columns and col_s not in test_df.columns:
            test_df[col_s] = test_df[col_raw].apply(_to_seconds)

    return test_df, test_map, None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--dataset", required=True, choices=["iemocap", "meld"])

    ap.add_argument("--iemocap_root", default=None)

    ap.add_argument("--train_csv",      default=None)
    ap.add_argument("--dev_csv",        default=None)
    ap.add_argument("--test_csv",       default=None)
    ap.add_argument("--test_video_dir", default=None)

    ap.add_argument("--out", required=True)

    ap.add_argument("--visual_mode", required=True,
                    choices=["vit", "landmarks", "landmarks_exp", "au"])
    ap.add_argument("--au_model",    default="svm",
                    choices=["svm", "xgb", "rf", "jaanet", "drml"],
                    help="[au] py-feat AU detector model.")
    ap.add_argument("--model_id",    default="dima806/facial_emotions_image_detection")
    ap.add_argument("--device",      default="cuda")
    ap.add_argument("--frames",      type=int,   default=16)
    ap.add_argument("--face_margin", type=float, default=0.2)
    ap.add_argument("--fullframe_fallback",   action="store_true", default=False)
    ap.add_argument("--scene_cut_threshold",  type=float, default=0.3)
    ap.add_argument("--tddfa_root",           default="/home/Imatge/oriol/unimodal/visual/3DDFA_V2")
    ap.add_argument("--tddfa_onnx_path",      default="/home/Imatge/oriol/unimodal/visual/mb1_120x120.onnx")
    ap.add_argument("--asd_weights",          default=None)
    ap.add_argument("--asd_threshold",        type=float, default=0.5)
    ap.add_argument("--asd_iou_threshold",    type=float, default=0.5)
    ap.add_argument("--asd_max_seconds",      type=float, default=60.0)
    ap.add_argument("--missing_policy",       default="zeros",
                    choices=["zeros", "error"])

    ap.add_argument("--degradation", required=True, choices=["blur", "occlusion"])
    ap.add_argument("--blur_sigma",   type=float, default=3.0,
                    help="[blur] Gaussian blur sigma in pixels. Sweep: 1 3 7 15")
    ap.add_argument("--occlude_frac", type=float, default=0.5,
                    help="[occlusion] Fraction of face box height to black out. "
                         "Applied to the DETECTED face box, not the scene. "
                         "Sweep: 0.25 0.5 0.75 1.0")
    args = ap.parse_args()

    if args.dataset == "iemocap" and not args.iemocap_root:
        ap.error("--dataset iemocap requires --iemocap_root")
    if args.dataset == "meld" and not all(
        [args.train_csv, args.dev_csv, args.test_csv, args.test_video_dir]
    ):
        ap.error("--dataset meld requires --train_csv, --dev_csv, --test_csv, --test_video_dir")

    tag = (f"{args.visual_mode}_blur_s{args.blur_sigma}"
           if args.degradation == "blur"
           else f"{args.visual_mode}_occlude{int(args.occlude_frac * 100)}pct")
    print(f"Dataset: {args.dataset}  |  Degradation: {tag}")
    if args.degradation == "occlusion":
        print("  Occlusion is FACE-AWARE: applied to the detected face box, "
              "not a fixed scene region.")

    # ── frame/extraction setup ─────────────────────────────────────────────────
    if args.degradation == "blur":
        frame_transform = make_blur_transform(args.blur_sigma)
    else:
        frame_transform = None  # occlusion handled per-frame inside extraction

    # ── load data ──────────────────────────────────────────────────────────────
    if args.dataset == "iemocap":
        _, _, test_df, id_map = load_iemocap_visual_dataframes(args.iemocap_root)
        test_ids = sorted(test_df["Dialogue_ID"].astype(int).unique())
        test_map = {d: d for d in test_ids}
    else:
        test_df, test_map, id_map = _load_meld_test_visual(
            args.train_csv, args.dev_csv, args.test_csv, args.test_video_dir
        )
    print(f"Test utterances: {len(test_df)}")

    # ── load models ────────────────────────────────────────────────────────────
    if args.visual_mode == "au":
        cfg = AUConfig(
            device=args.device,
            au_model=args.au_model,
            num_frames=args.frames,
            face_margin=args.face_margin,
            use_fullframe_fallback=args.fullframe_fallback,
            scene_cut_threshold=args.scene_cut_threshold,
            asd_weights=args.asd_weights,
            asd_threshold=args.asd_threshold,
            asd_iou_threshold=args.asd_iou_threshold,
            asd_max_seconds=args.asd_max_seconds,
        )
        cfg.visual_mode = "au"   # tag for dispatch functions below
        mtcnn, feat_detector, embed_dim, asd = load_au_models(cfg)
        processor, model = None, feat_detector
    else:
        cfg = VisualConfig(
            visual_mode=args.visual_mode,
            model_id=args.model_id,
            device=args.device,
            num_frames=args.frames,
            face_margin=args.face_margin,
            use_fullframe_fallback=args.fullframe_fallback,
            scene_cut_threshold=args.scene_cut_threshold,
            tddfa_root=args.tddfa_root,
            tddfa_onnx_path=args.tddfa_onnx_path,
            asd_weights=args.asd_weights,
            asd_threshold=args.asd_threshold,
            asd_iou_threshold=args.asd_iou_threshold,
            asd_max_seconds=args.asd_max_seconds,
        )
        mtcnn, processor, model, embed_dim, asd = load_visual_models(cfg)
    print(f"embed_dim: {embed_dim}")

    # ── extract ────────────────────────────────────────────────────────────────
    out: dict = {}
    n_ok = n_missing = 0

    for _, row in tqdm(test_df.iterrows(), total=len(test_df),
                       desc=f"Degraded visual ({tag})"):
        d_orig    = int(row["Dialogue_ID"])
        u         = int(row["Utterance_ID"])
        d_new     = test_map[d_orig]
        vid_path  = str(row.get("Video_Path", ""))
        if args.dataset == "meld":
            start_sec = None
            end_sec = None
        else:
            start_sec = float(row["Start_Time"]) if "Start_Time" in test_df.columns else None
            end_sec   = float(row["End_Time"])   if "End_Time"   in test_df.columns else None

        if not vid_path or not os.path.exists(vid_path):
            if args.missing_policy == "error":
                raise FileNotFoundError(vid_path)
            out.setdefault(d_new, {})[u] = np.zeros(embed_dim, dtype=np.float32)
            n_missing += 1
            continue

        try:
            if args.degradation == "blur":
                emb = _extract_blur(
                    vid_path, mtcnn, processor, model, cfg, embed_dim,
                    asd, start_sec, end_sec, frame_transform
                )
            else:
                emb = _extract_occlusion(
                    vid_path, mtcnn, processor, model, cfg, embed_dim,
                    asd, start_sec, end_sec, args.occlude_frac
                )
        except Exception as e:
            print(f"  [WARN] {vid_path}: {e}")
            out.setdefault(d_new, {})[u] = np.zeros(embed_dim, dtype=np.float32)
            n_missing += 1
            continue

        out.setdefault(d_new, {})[u] = np.asarray(emb)
        n_ok += 1

    out = {d: dict(sorted(u_m.items())) for d, u_m in sorted(out.items())}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    id_map_path = args.out.replace(".pkl", "_id_map.pkl")
    with open(id_map_path, "wb") as f:
        pickle.dump(id_map, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved {sum(len(v) for v in out.values())} utterances → {args.out}")
    print(f"OK: {n_ok}  Missing/failed: {n_missing}")


if __name__ == "__main__":
    main()