import io
import os
import re
import time
import pickle
import argparse
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from models_au import (
    AUConfig,
    load_au_models,
    extract_utterance_au_embedding,
    load_iemocap_au_dataframes,
)


# ── MELD CSV loader ───────────────────────────────────────────────────────────

def _fix_meld_mojibake(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    def _rc(m: re.Match) -> str:
        b = ord(m.group(0))
        try:    return bytes([b]).decode("windows-1252")
        except: return m.group(0)
    return re.sub(r"[\u0080-\u009f]", _rc, text).replace("\xa0", " ")


def load_meld_csv(path: str) -> pd.DataFrame:
    """Load a MELD CSV, repairing the Windows-1252 mojibake encoding bug."""
    with open(path, "rb") as f:
        raw = f.read()
    return pd.read_csv(io.StringIO(_fix_meld_mojibake(raw)))


def video_path_from_ids(video_dir: str,
                        dialogue_id: int, utterance_id: int) -> str:
    return os.path.join(video_dir, f"dia{dialogue_id}_utt{utterance_id}.mp4")


# ── Per-split extraction ──────────────────────────────────────────────────────

def extract_split(
    visual:         dict,
    df:             pd.DataFrame,
    d_map:          dict,
    video_dir:      str,
    mtcnn,
    feat_detector,
    cfg:            AUConfig,
    embed_dim:      int,
    missing_policy: str,
    asd,
    desc:           str,
) -> Tuple[int, int, float]:
    """
    Extract AU embeddings for one split (train / dev / test).
    Stores results in-place: visual[new_dialogue_id][utterance_id] = embedding.
    Returns (n_missing, n_rows, total_inference_seconds).
    """
    df = df.copy()
    df["Dialogue_ID"]  = df["Dialogue_ID"].astype(int)
    df["Utterance_ID"] = df["Utterance_ID"].astype(int)

    missing         = 0
    inference_times = []

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
            visual.setdefault(new_d, {})[u] = np.zeros(
                (embed_dim,), dtype=np.float32)
            continue

        t0  = time.perf_counter()
        emb = extract_utterance_au_embedding(
            video_path=video_path,
            mtcnn=mtcnn,
            feat_detector=feat_detector,
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
        visual.setdefault(new_d, {})[u] = emb

    total_s = sum(inference_times)
    return missing, len(df), total_s


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Extract Action Unit embeddings for MELD / IEMOCAP."
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    ap.add_argument("--dataset", choices=["meld", "iemocap"], required=True)

    # MELD
    ap.add_argument("--train_csv",        default=None)
    ap.add_argument("--dev_csv",          default=None)
    ap.add_argument("--test_csv",         default=None)
    ap.add_argument("--train_video_dir",  default=None)
    ap.add_argument("--dev_video_dir",    default=None)
    ap.add_argument("--test_video_dir",   default=None)

    # IEMOCAP
    ap.add_argument("--iemocap_root", default=None,
                    help="Path to IEMOCAP_full_release directory.")

    # Output
    ap.add_argument("--out", required=True,
                    help="Output pkl path (e.g. data/iemocap_action_units.pkl)")

    # AU settings
    ap.add_argument("--au_model", default="svm",
                    choices=["svm", "xgb", "rf", "jaanet", "drml"],
                    help="py-feat AU detector model (default: svm).")

    # Shared extraction settings
    ap.add_argument("--frames",              type=int,   default=16)
    ap.add_argument("--face_margin",         type=float, default=0.2)
    ap.add_argument("--fullframe_fallback",  action="store_true", default=False)
    ap.add_argument("--scene_cut_threshold", type=float, default=0.3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # ASD
    ap.add_argument("--asd_weights",       default=None)
    ap.add_argument("--asd_threshold",     type=float, default=0.5)
    ap.add_argument("--asd_iou_threshold", type=float, default=0.5)
    ap.add_argument("--asd_max_seconds",   type=float, default=60.0)

    # Misc
    ap.add_argument("--missing_policy", choices=["zeros", "error"], default="zeros")
    ap.add_argument("--limit",    type=int, default=0,
                    help="Limit rows per split (for quick tests).")
    ap.add_argument("--shard",    type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)

    args = ap.parse_args()

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

    print("Configuration:")
    print(f"  dataset    : {args.dataset}")
    print(f"  au_model   : {cfg.au_model}")
    print(f"  num_frames : {cfg.num_frames}")
    print(f"  face_margin: {cfg.face_margin}")
    print(f"  device     : {cfg.device}")
    print(f"  ASD        : {'enabled → ' + str(cfg.asd_weights) if cfg.asd_weights else 'disabled'}")
    print()

    # ── Load models ───────────────────────────────────────────────────────────
    mtcnn, feat_detector, embed_dim, asd = load_au_models(cfg)
    print(f"  embed_dim  : {embed_dim}  (pass --D_visual {embed_dim} to train.py)\n")

    # ── Load dataframes ───────────────────────────────────────────────────────
    id_map = None

    if args.dataset == "meld":
        if not all([args.train_csv, args.dev_csv, args.test_csv,
                    args.train_video_dir, args.dev_video_dir,
                    args.test_video_dir]):
            ap.error("--dataset meld requires "
                     "--train/dev/test_csv and --train/dev/test_video_dir")
        train_df = load_meld_csv(args.train_csv)
        dev_df   = load_meld_csv(args.dev_csv)
        test_df  = load_meld_csv(args.test_csv)
        train_video_dir = args.train_video_dir
        dev_video_dir   = args.dev_video_dir
        test_video_dir  = args.test_video_dir

    else:  # iemocap
        if not args.iemocap_root:
            ap.error("--dataset iemocap requires --iemocap_root")
        train_df, dev_df, test_df, id_map = load_iemocap_au_dataframes(
            args.iemocap_root)
        train_video_dir = dev_video_dir = test_video_dir = ""

    # ── Limit rows (for testing) ──────────────────────────────────────────────
    if args.limit and args.limit > 0:
        train_df = train_df.head(args.limit)
        dev_df   = dev_df.head(args.limit)
        test_df  = test_df.head(args.limit)
        print(f"[limit={args.limit}] restricted each split to first {args.limit} rows.")

    # ── Sharding ──────────────────────────────────────────────────────────────
    if args.n_shards > 1:
        all_diags = sorted(
            set(train_df["Dialogue_ID"].astype(int).tolist() +
                dev_df["Dialogue_ID"].astype(int).tolist() +
                test_df["Dialogue_ID"].astype(int).tolist())
        )
        shard_diags = set(all_diags[args.shard::args.n_shards])
        train_df = train_df[train_df["Dialogue_ID"].astype(int).isin(shard_diags)].reset_index(drop=True)
        dev_df   = dev_df[dev_df["Dialogue_ID"].astype(int).isin(shard_diags)].reset_index(drop=True)
        test_df  = test_df[test_df["Dialogue_ID"].astype(int).isin(shard_diags)].reset_index(drop=True)
        print(f"Shard {args.shard}/{args.n_shards}: "
              f"{len(train_df)} train + {len(dev_df)} dev + {len(test_df)} test rows")

    # ── Build dialogue ID maps (same logic as main_landmarks_new) ─────────────
    train_ids = sorted(train_df["Dialogue_ID"].astype(int).unique())
    dev_ids   = sorted(dev_df["Dialogue_ID"].astype(int).unique())
    test_ids  = sorted(test_df["Dialogue_ID"].astype(int).unique())

    train_map   = {d: d for d in train_ids}
    offset_dev  = (max(train_ids) + 1) if train_ids else 0
    dev_map     = {d: offset_dev + i for i, d in enumerate(dev_ids)}
    offset_test = (max(dev_map.values()) + 1) if dev_map else offset_dev
    test_map    = {d: offset_test + i for i, d in enumerate(test_ids)}

    print("ID ranges (new):")
    if train_ids:
        print(f"  train: ({min(train_map.values())}, {max(train_map.values())})  count={len(train_map)}")
    if dev_ids:
        print(f"  dev:   ({min(dev_map.values())},   {max(dev_map.values())})   count={len(dev_map)}")
    if test_ids:
        print(f"  test:  ({min(test_map.values())},  {max(test_map.values())})  count={len(test_map)}")
    print()

    # ── Extract ───────────────────────────────────────────────────────────────
    visual  = {}
    t_start = time.perf_counter()

    miss_train, n_train, s_train = extract_split(
        visual, train_df, train_map, train_video_dir,
        mtcnn, feat_detector, cfg, embed_dim, args.missing_policy, asd,
        desc="train",
    )
    miss_dev, n_dev, s_dev = extract_split(
        visual, dev_df, dev_map, dev_video_dir,
        mtcnn, feat_detector, cfg, embed_dim, args.missing_policy, asd,
        desc="dev",
    )
    miss_test, n_test, s_test = extract_split(
        visual, test_df, test_map, test_video_dir,
        mtcnn, feat_detector, cfg, embed_dim, args.missing_policy, asd,
        desc="test",
    )

    total_wall = time.perf_counter() - t_start

    # Sort keys for deterministic pkl
    visual = {
        d: dict(sorted(u_map.items()))
        for d, u_map in sorted(visual.items())
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(visual, f, protocol=pickle.HIGHEST_PROTOCOL)

    if id_map is not None:
        id_map_path = args.out.replace(".pkl", "_id_map.pkl")
        with open(id_map_path, "wb") as f:
            pickle.dump(id_map, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved id_map → {id_map_path}")


if __name__ == "__main__":
    main()