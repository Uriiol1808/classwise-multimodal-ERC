import os
import re
import io
import time
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Optional, Tuple

from .models_landmarks_new import (
    VisualConfig,
    load_visual_models,
    load_iemocap_visual_dataframes,
    extract_utterance_visual_embedding,
    extract_utterance_landmark_embedding,
)


# ── Encoding fix (same mojibake bug as all MELD CSVs) ─────────────────────────

def _fix_meld_mojibake(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    def _rc(m: re.Match) -> str:
        b = ord(m.group(0))
        try:    return bytes([b]).decode("windows-1252")
        except: return m.group(0)
    return re.sub(r"[\u0080-\u009f]", _rc, text).replace("\xa0", " ")

def load_csv(path: str) -> pd.DataFrame:
    """Loads a MELD CSV, repairing the Windows-1252 mojibake encoding bug."""
    with open(path, "rb") as f:
        raw = f.read()
    return pd.read_csv(io.StringIO(_fix_meld_mojibake(raw)))


# ── Helpers ───────────────────────────────────────────────────────────────────

def video_path_from_ids(video_dir: str, dialogue_id: int, utterance_id: int) -> str:
    return os.path.join(video_dir, f"dia{dialogue_id}_utt{utterance_id}.mp4")


def add_split_visual(
    visual: dict,
    df: pd.DataFrame,
    d_map: dict,
    video_dir: str,
    mtcnn,
    processor,
    model,
    cfg: VisualConfig,
    embed_dim: int,
    missing_policy: str,
    asd,
    desc: str,
) -> Tuple[int, int, float, float]:
    """
    Stores embeddings under new dialogue IDs: visual[new_d][u] = embedding.
    Video lookup uses original Dialogue_ID for filenames.

    Returns: (missing, n_rows, total_inference_s, avg_ms_per_utt)
    """
    required = ["Dialogue_ID", "Utterance_ID"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"{desc}: missing column '{c}'. Columns: {list(df.columns)}")

    df = df.copy()
    df["Dialogue_ID"] = df["Dialogue_ID"].astype(int)
    df["Utterance_ID"] = df["Utterance_ID"].astype(int)

    missing          = 0
    inference_times: list = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        d_orig = int(row["Dialogue_ID"])
        u      = int(row["Utterance_ID"])
        new_d  = d_map[d_orig]

        video_path = (
            row["Video_Path"] if "Video_Path" in df.columns
            else video_path_from_ids(video_dir, d_orig, u)
        )
        start_sec = float(row["Start_Time"]) if "Start_Time" in df.columns else None
        end_sec   = float(row["End_Time"])   if "End_Time"   in df.columns else None

        if not os.path.exists(video_path):
            missing += 1
            if missing_policy == "error":
                raise FileNotFoundError(video_path)
            visual.setdefault(new_d, {})[u] = np.zeros((embed_dim,), dtype=np.float32)
            continue

        t0 = time.perf_counter()

        if cfg.visual_mode in ("landmarks", "landmarks_exp"):
            emb = extract_utterance_landmark_embedding(
                video_path=video_path,
                mtcnn=mtcnn,
                tddfa=model,
                embed_dim=embed_dim,
                num_frames=cfg.num_frames,
                face_margin=cfg.face_margin,
                use_fullframe_fallback=cfg.use_fullframe_fallback,
                scene_cut_threshold=cfg.scene_cut_threshold,
                asd=asd,
                asd_threshold=cfg.asd_threshold,
                asd_iou_threshold=cfg.asd_iou_threshold,
                asd_max_seconds=cfg.asd_max_seconds,
                start_sec=start_sec,
                end_sec=end_sec,
                visual_mode=cfg.visual_mode,
            )
        else:
            emb = extract_utterance_visual_embedding(
                video_path=video_path,
                mtcnn=mtcnn,
                processor=processor,
                model=model,
                device=cfg.device,
                embed_dim=embed_dim,
                num_frames=cfg.num_frames,
                face_margin=cfg.face_margin,
                use_fullframe_fallback=cfg.use_fullframe_fallback,
                scene_cut_threshold=cfg.scene_cut_threshold,
                asd=asd,
                asd_threshold=cfg.asd_threshold,
                asd_iou_threshold=cfg.asd_iou_threshold,
                asd_max_seconds=cfg.asd_max_seconds,
                start_sec=start_sec,
                end_sec=end_sec,
            )
        inference_times.append(time.perf_counter() - t0)
        visual.setdefault(new_d, {})[u] = np.asarray(emb)

    total_s = sum(inference_times)
    n_rows  = len(df)
    avg_ms  = (total_s / n_rows * 1000) if n_rows > 0 else 0.0
    return missing, n_rows, total_s, avg_ms


def main():
    ap = argparse.ArgumentParser(
        description="Extract visual embeddings (ViT or 3D landmarks) for MELD/IEMOCAP."
    )

    # ── dataset ───────────────────────────────────────────────────────────────
    ap.add_argument("--dataset", choices=["meld", "iemocap"], default="meld")
    # MELD args
    ap.add_argument("--train_csv",       default=None)
    ap.add_argument("--dev_csv",         default=None)
    ap.add_argument("--test_csv",        default=None)
    ap.add_argument("--train_video_dir", default=None)
    ap.add_argument("--dev_video_dir",   default=None)
    ap.add_argument("--test_video_dir",  default=None)
    # IEMOCAP args
    ap.add_argument("--iemocap_root", default=None,
                    help="Path to IEMOCAP_full_release directory.")

    ap.add_argument("--out", required=True)

    # ── visual mode ───────────────────────────────────────────────────────────
    ap.add_argument("--visual_mode", choices=["vit", "landmarks", "landmarks_exp"], default="vit",
                    help="vit: ViT appearance features (default). "
                         "landmarks: 3DDFA-V2 raw 3D landmarks, mean+std → 408-d. "
                         "landmarks_exp: 3DDFA-V2 expression coefficients + pose, mean+std+delta → 39-d.")

    # ── ViT settings (ignored when --visual_mode landmarks) ───────────────────
    ap.add_argument("--model_id",    default="dima806/facial_emotions_image_detection")

    # ── 3DDFA-V2 settings (ignored when --visual_mode vit) ────────────────────
    ap.add_argument("--tddfa_root", default="",
                    help="Path to cloned https://github.com/cleardusk/3DDFA_V2")
    ap.add_argument("--tddfa_onnx_path", default="",
                    help="Path to .onnx weights file (e.g. mb1_120x120.onnx)")

    # ── shared settings ───────────────────────────────────────────────────────
    ap.add_argument("--frames",      type=int,   default=16)
    ap.add_argument("--face_margin", type=float, default=0.2)
    ap.add_argument("--fullframe_fallback", action="store_true", default=False)
    ap.add_argument("--scene_cut_threshold", type=float, default=0.3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # ── ASD ───────────────────────────────────────────────────────────────────
    ap.add_argument("--asd_weights",       default=None)
    ap.add_argument("--asd_threshold",     type=float, default=0.5)
    ap.add_argument("--asd_iou_threshold", type=float, default=0.5)
    ap.add_argument("--asd_max_seconds",   type=float, default=60.0)

    ap.add_argument("--missing_policy", choices=["zeros", "error"], default="zeros")
    ap.add_argument("--limit", type=int, default=0)

    ap.add_argument("--shard", type=int, default=0,
                help="Shard index (0-based)")
    ap.add_argument("--n_shards", type=int, default=1,
                    help="Total number of shards")

    args = ap.parse_args()
    limit_rows = None if not args.limit or args.limit <= 0 else int(args.limit)

    cfg = VisualConfig(
        device=args.device,
        visual_mode=args.visual_mode,
        model_id=args.model_id,
        tddfa_root=args.tddfa_root,
        tddfa_onnx_path=args.tddfa_onnx_path,
        num_frames=args.frames,
        face_margin=args.face_margin,
        use_fullframe_fallback=args.fullframe_fallback,
        scene_cut_threshold=args.scene_cut_threshold,
        asd_weights=args.asd_weights,
        asd_threshold=args.asd_threshold,
        asd_iou_threshold=args.asd_iou_threshold,
        asd_max_seconds=args.asd_max_seconds,
    )

    print("Configuration:")
    print(f"  dataset             : {args.dataset}")
    print(f"  visual_mode         : {cfg.visual_mode}")
    if cfg.visual_mode == "vit":
        print(f"  model_id            : {cfg.model_id}")
    else:
        print(f"  tddfa_root          : {cfg.tddfa_root}")
        print(f"  tddfa_onnx_path     : {cfg.tddfa_onnx_path}")
    print(f"  num_frames          : {cfg.num_frames}")
    print(f"  face_margin         : {cfg.face_margin}")
    print(f"  fullframe_fallback  : {cfg.use_fullframe_fallback}")
    print(f"  scene_cut_threshold : {cfg.scene_cut_threshold}")
    print(f"  ASD                 : {'enabled → ' + str(cfg.asd_weights) if cfg.asd_weights else 'disabled'}")
    print(f"  device              : {cfg.device}")
    print()

    mtcnn, processor, model, embed_dim, asd = load_visual_models(cfg)
    print(f"  embed_dim           : {embed_dim}")
    print()

    # ── load data ─────────────────────────────────────────────────────────────
    id_map = None

    if args.dataset == "meld":
        if not all([args.train_csv, args.dev_csv, args.test_csv,
                    args.train_video_dir, args.dev_video_dir, args.test_video_dir]):
            ap.error("--dataset meld requires --train/dev/test_csv and --train/dev/test_video_dir")
        train_df = load_csv(args.train_csv)
        dev_df   = load_csv(args.dev_csv)
        test_df  = load_csv(args.test_csv)
        train_video_dir = args.train_video_dir
        dev_video_dir   = args.dev_video_dir
        test_video_dir  = args.test_video_dir
    else:  # iemocap
        if not args.iemocap_root:
            ap.error("--dataset iemocap requires --iemocap_root")
        train_df, dev_df, test_df, id_map = load_iemocap_visual_dataframes(args.iemocap_root)
        train_video_dir = dev_video_dir = test_video_dir = ""

    if limit_rows:
        train_df = train_df.head(limit_rows)
        dev_df   = dev_df.head(limit_rows)
        test_df  = test_df.head(limit_rows)

    if args.n_shards > 1:
        # split unique dialogue IDs across shards
        all_diag = sorted(train_df["Dialogue_ID"].astype(int).unique().tolist() +
                        dev_df["Dialogue_ID"].astype(int).unique().tolist() +
                        test_df["Dialogue_ID"].astype(int).unique().tolist())
        shard_diags = set(all_diag[args.shard::args.n_shards])
        train_df = train_df[train_df["Dialogue_ID"].astype(int).isin(shard_diags)].reset_index(drop=True)
        dev_df   = dev_df[dev_df["Dialogue_ID"].astype(int).isin(shard_diags)].reset_index(drop=True)
        test_df  = test_df[test_df["Dialogue_ID"].astype(int).isin(shard_diags)].reset_index(drop=True)
        print(f"Shard {args.shard}/{args.n_shards}: {len(train_df)} train + {len(dev_df)} dev + {len(test_df)} test rows")

    train_ids = sorted(train_df["Dialogue_ID"].astype(int).unique())
    dev_ids   = sorted(dev_df["Dialogue_ID"].astype(int).unique())
    test_ids  = sorted(test_df["Dialogue_ID"].astype(int).unique())

    train_map   = {d: d for d in train_ids}
    offset_dev  = (max(train_ids) + 1) if train_ids else 0
    dev_map     = {d: offset_dev + i for i, d in enumerate(dev_ids)}
    offset_test = (max(dev_map.values()) + 1) if dev_map else offset_dev
    test_map    = {d: offset_test + i for i, d in enumerate(test_ids)}

    print("Original Dialogue_ID overlap:", {
        "train∩dev":  len(set(train_ids) & set(dev_ids)),
        "train∩test": len(set(train_ids) & set(test_ids)),
        "dev∩test":   len(set(dev_ids)   & set(test_ids)),
    })
    print("ID ranges (new):")
    print("  train:", (min(train_map.values()), max(train_map.values())), "count:", len(train_map))
    print("  test: ", (min(test_map.values()),  max(test_map.values())),  "count:", len(test_map))
    print()

    visual = {}
    t_start = time.perf_counter()

    miss_train, n_train, s_train, ms_train = add_split_visual(
        visual, train_df, train_map, train_video_dir,
        mtcnn, processor, model, cfg, embed_dim, args.missing_policy, asd,
        desc="Extracting visual (train)",
    )
    miss_dev, n_dev, s_dev, ms_dev = add_split_visual(
        visual, dev_df, dev_map, dev_video_dir,
        mtcnn, processor, model, cfg, embed_dim, args.missing_policy, asd,
        desc="Extracting visual (dev)",
    )
    miss_test, n_test, s_test, ms_test = add_split_visual(
        visual, test_df, test_map, test_video_dir,
        mtcnn, processor, model, cfg, embed_dim, args.missing_policy, asd,
        desc="Extracting visual (test)",
    )

    total_wall = time.perf_counter() - t_start

    visual = {
        d: dict(sorted(u_map.items()))
        for d, u_map in sorted(visual.items())
    }

    with open(args.out, "wb") as f:
        pickle.dump(visual, f, protocol=pickle.HIGHEST_PROTOCOL)

    if id_map is not None:
        id_map_path = args.out.replace(".pkl", "_id_map.pkl")
        with open(id_map_path, "wb") as f:
            pickle.dump(id_map, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved id_map → {id_map_path}")

    n_dialogues = len(visual)
    n_utts      = sum(len(u_map) for u_map in visual.values())

    print("\nSaved:", args.out)
    print("Processed rows:", {"train": n_train, "dev": n_dev, "test": n_test})
    print(f"Dialogues: {n_dialogues}  Utterances: {n_utts}  Dim: {embed_dim}")

    miss_total = miss_train + miss_dev + miss_test
    if miss_total:
        print("Missing videos:", {"train": miss_train, "dev": miss_dev,
                                  "test": miss_test, "total": miss_total})

    # ── Timing & memory report ─────────────────────────────────────────────────
    total_inf  = s_train + s_dev + s_test
    total_utts = n_train + n_dev + n_test
    avg_ms_all = (total_inf / total_utts * 1000) if total_utts > 0 else 0.0

    is_cuda = cfg.device.startswith("cuda") and torch.cuda.is_available()

    if is_cuda:
        dev_idx      = torch.cuda.current_device()
        gpu_peak_mb  = torch.cuda.max_memory_allocated(dev_idx) / 1024**2
        gpu_res_mb   = torch.cuda.memory_reserved(dev_idx)      / 1024**2
        gpu_total_mb = torch.cuda.get_device_properties(dev_idx).total_memory / 1024**2

    cpu_rss_mb: Optional[float] = None
    try:
        import psutil
        cpu_rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    except ImportError:
        pass

    if cfg.visual_mode == "vit":
        model_params = sum(p.numel() for p in model.parameters())
        gpu_model_mb = model_params * 2 / 1024**2
        cpu_model_mb = model_params * 4 / 1024**2
        print(f"  Model weights (fp16): {gpu_model_mb:.0f} MB  (GPU)")
        print(f"  Model weights (fp32): {cpu_model_mb:.0f} MB  (CPU equivalent)")
    else:
        print(f"  Model: 3DDFA-V2 ONNX (no parameter count — not a PyTorch module)")

    W = 54
    print()
    print("─" * W)
    print(f"  {'Timing & memory report':^{W-4}}")
    print("─" * W)
    print(f"  {'Split':<8}  {'utts':>6}  {'total':>8}  {'ms/utt':>8}")
    print(f"  {'─'*6:<8}  {'─'*6:>6}  {'─'*8:>8}  {'─'*8:>8}")
    print(f"  {'train':<8}  {n_train:>6}  {s_train:>7.1f}s  {ms_train:>7.2f}ms")
    print(f"  {'dev':<8}  {n_dev:>6}  {s_dev:>7.1f}s  {ms_dev:>7.2f}ms")
    print(f"  {'test':<8}  {n_test:>6}  {s_test:>7.1f}s  {ms_test:>7.2f}ms")
    print(f"  {'─'*6:<8}  {'─'*6:>6}  {'─'*8:>8}  {'─'*8:>8}")
    print(f"  {'total':<8}  {total_utts:>6}  {total_inf:>7.1f}s  {avg_ms_all:>7.2f}ms")
    print(f"  wall time: {total_wall:.1f}s  (video decode + face detect + inference)")
    print()

    if is_cuda:
        print(f"  GPU  : {torch.cuda.get_device_name(dev_idx)}")
        print(f"  Peak alloc  : {gpu_peak_mb:7.0f} MB  /  {gpu_total_mb:.0f} MB  "
              f"({gpu_peak_mb / gpu_total_mb * 100:.1f}%)")
        print(f"  Reserved    : {gpu_res_mb:7.0f} MB")
    else:
        print(f"  Device: CPU")
        print(f"  Tip   : re-run with --limit 50 --device cpu for a fast CPU baseline.")

    if cpu_rss_mb is not None:
        print(f"  CPU RSS     : {cpu_rss_mb:7.0f} MB  (process peak working set)")
    else:
        print(f"  CPU RSS     : n/a  (install psutil for CPU memory tracking)")

    print("─" * W)


if __name__ == "__main__":
    main()