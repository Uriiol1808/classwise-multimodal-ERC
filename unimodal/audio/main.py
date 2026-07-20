from __future__ import annotations

import os
import time
import argparse
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm

from .models import (
    AudioConfig,
    load_csv,
    load_iemocap_audio_dataframes,
    load_audio_model,
    get_model_weight_mb,
    load_wav,
    embed_audio_batch,
)


def audio_path_from_ids(audio_dir: str, dialogue_id: int, utterance_id: int, audio_ext: str) -> str:
    return os.path.join(audio_dir, f"dia{dialogue_id}_utt{utterance_id}{audio_ext}")


def add_split_audio(
    out_nested: Dict[int, Dict[int, np.ndarray]],
    df,
    d_map: Dict[int, int],
    audio_dir: str,
    audio_ext: str,
    cfg: AudioConfig,
    processor,
    model,
    embed_dim: int,
    missing_policy: str,
    desc: str,
) -> Tuple[int, int, int, float, float]:
    """
    Fills out_nested[new_d][u] = embedding.

    Returns: (missing, silent, n_rows, total_inference_s, avg_ms_per_utt)
    """
    required = ["Dialogue_ID", "Utterance_ID"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"{desc}: missing column '{c}'. Columns: {list(df.columns)}")

    df = df.copy()
    df["Dialogue_ID"] = df["Dialogue_ID"].astype(int)
    df["Utterance_ID"] = df["Utterance_ID"].astype(int)

    missing = 0
    silent  = 0
    batch_meta:   List[Tuple[int, int]] = []
    batch_wavs:   List[np.ndarray]      = []
    batch_silent: List[bool]            = []
    inference_times: List[float]        = []
    inference_utts:  List[int]          = []

    def flush():
        if not batch_wavs:
            return
        t0 = time.perf_counter()
        E = embed_audio_batch(
            batch_wavs, processor=processor, model=model, device=cfg.device,
            sampling_rate=cfg.target_sr, pooling=cfg.pooling,
            silent_mask=batch_silent, embed_dim=embed_dim,
        )
        inference_times.append(time.perf_counter() - t0)
        inference_utts.append(len(batch_wavs))

        for (new_d, u), e in zip(batch_meta, E):
            out_nested.setdefault(new_d, {})[u] = np.asarray(e)

        batch_meta.clear()
        batch_wavs.clear()
        batch_silent.clear()

    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        d_orig = int(row["Dialogue_ID"])
        u      = int(row["Utterance_ID"])
        new_d  = d_map[d_orig]

        apath = (
            row["Audio_Path"] if "Audio_Path" in df.columns
            else audio_path_from_ids(audio_dir, d_orig, u, audio_ext)
        )

        if not os.path.exists(apath):
            missing += 1
            if missing_policy == "error":
                raise FileNotFoundError(apath)
            out_nested.setdefault(new_d, {})[u] = np.zeros((embed_dim,), dtype=np.float32)
            continue

        try:
            wav, _sr, is_silent = load_wav(apath, target_sr=cfg.target_sr, max_seconds=cfg.max_seconds)
        except Exception:
            missing += 1
            if missing_policy == "error":
                raise
            out_nested.setdefault(new_d, {})[u] = np.zeros((embed_dim,), dtype=np.float32)
            continue

        if is_silent:
            silent += 1

        batch_meta.append((new_d, u))
        batch_wavs.append(wav)
        batch_silent.append(is_silent)

        if len(batch_wavs) >= cfg.batch_size:
            flush()

    flush()

    total_s = sum(inference_times)
    total_u = sum(inference_utts)
    avg_ms  = (total_s / total_u * 1000) if total_u > 0 else 0.0
    return missing, silent, len(df), total_s, avg_ms


def main():
    ap = argparse.ArgumentParser()

    # ── dataset source ────────────────────────────────────────────────────────
    ap.add_argument("--dataset", choices=["meld", "iemocap"], default="meld")
    # MELD args
    ap.add_argument("--train_csv",       default=None)
    ap.add_argument("--dev_csv",         default=None)
    ap.add_argument("--test_csv",        default=None)
    ap.add_argument("--train_audio_dir", default=None)
    ap.add_argument("--dev_audio_dir",   default=None)
    ap.add_argument("--test_audio_dir",  default=None)
    ap.add_argument("--audio_ext",       default=".wav")
    # IEMOCAP args
    ap.add_argument("--iemocap_root", default=None,
                    help="Path to IEMOCAP_full_release directory.")
    ap.add_argument("--out",             required=True)

    ap.add_argument("--model_id",    default="iic/emotion2vec_plus_large")
    ap.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--target_sr",   type=int,   default=16000)
    ap.add_argument("--batch_size",  type=int,   default=8)
    ap.add_argument("--max_seconds", type=float, default=10.0, help="0 = no truncation")

    ap.add_argument("--missing_policy", choices=["zeros", "error"], default="zeros")

    ap.add_argument("--limit", type=int, default=0)

    args = ap.parse_args()

    limit_rows = None if args.limit <= 0 else int(args.limit)

    cfg = AudioConfig(
        device=args.device,
        model_id=args.model_id,
        target_sr=args.target_sr,
        max_seconds=(args.max_seconds if args.max_seconds and args.max_seconds > 0 else None),
        batch_size=args.batch_size,
        pooling="mean",
    )

    print("Configuration:")
    print(f"  dataset        : {args.dataset}")
    print(f"  model_id       : {cfg.model_id}")
    print(f"  target_sr      : {cfg.target_sr}")
    print(f"  max_seconds    : {cfg.max_seconds}")
    print(f"  batch_size     : {cfg.batch_size}")
    print(f"  device         : {cfg.device}")
    print(f"  missing_policy : {args.missing_policy}")
    print()

    processor, model, embed_dim = load_audio_model(cfg)
    print(f"  embed_dim      : {embed_dim}")
    print()

    id_map = None

    if args.dataset == "meld":
        if not all([args.train_csv, args.dev_csv, args.test_csv,
                    args.train_audio_dir, args.dev_audio_dir, args.test_audio_dir]):
            ap.error("--dataset meld requires --train/dev/test_csv and --train/dev/test_audio_dir")
        train_df = load_csv(args.train_csv)
        dev_df   = load_csv(args.dev_csv)
        test_df  = load_csv(args.test_csv)
        train_audio_dir, dev_audio_dir, test_audio_dir = (
            args.train_audio_dir, args.dev_audio_dir, args.test_audio_dir)
    else:  # iemocap
        if not args.iemocap_root:
            ap.error("--dataset iemocap requires --iemocap_root")
        train_df, dev_df, test_df, id_map = load_iemocap_audio_dataframes(args.iemocap_root)
        # audio dirs unused for IEMOCAP (paths are in Audio_Path column)
        train_audio_dir = dev_audio_dir = test_audio_dir = ""

    if limit_rows:
        train_df = train_df.head(limit_rows)
        dev_df   = dev_df.head(limit_rows)
        test_df  = test_df.head(limit_rows)

    train_ids = sorted(train_df["Dialogue_ID"].astype(int).unique())
    dev_ids   = sorted(dev_df["Dialogue_ID"].astype(int).unique())
    test_ids  = sorted(test_df["Dialogue_ID"].astype(int).unique())

    train_map = {d: d for d in train_ids}
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
    print("  train:", (min(train_map.values()), max(train_map.values())) if train_map else None, "count:", len(train_map))
    print("  dev:  ", (min(dev_map.values()),   max(dev_map.values()))   if dev_map   else None, "count:", len(dev_map))
    print("  test: ", (min(test_map.values()),  max(test_map.values()))  if test_map  else None, "count:", len(test_map))
    print()

    out: Dict[int, Dict[int, np.ndarray]] = {}

    t_start = time.perf_counter()
    miss_train, silent_train, n_train, s_train, ms_train = add_split_audio(
        out, train_df, train_map, train_audio_dir, args.audio_ext,
        cfg, processor, model, embed_dim, args.missing_policy,
        desc=f"Extracting audio (train)",
    )
    miss_dev, silent_dev, n_dev, s_dev, ms_dev = add_split_audio(
        out, dev_df, dev_map, dev_audio_dir, args.audio_ext,
        cfg, processor, model, embed_dim, args.missing_policy,
        desc=f"Extracting audio (dev)",
    )
    miss_test, silent_test, n_test, s_test, ms_test = add_split_audio(
        out, test_df, test_map, test_audio_dir, args.audio_ext,
        cfg, processor, model, embed_dim, args.missing_policy,
        desc=f"Extracting audio (test)",
    )
    total_wall = time.perf_counter() - t_start

    out = {
        d: dict(sorted(u_map.items()))
        for d, u_map in sorted(out.items())
    }

    with open(args.out, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    if id_map is not None:
        id_map_path = args.out.replace(".pkl", "_id_map.pkl")
        with open(id_map_path, "wb") as f:
            pickle.dump(id_map, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved id_map → {id_map_path}")

    n_dialogues = len(out)
    n_utts      = sum(len(u_map) for u_map in out.values())
    any_d       = next(iter(out))
    any_u       = next(iter(out[any_d]))

    print("\nSaved:", args.out)
    print("Processed rows:", {"train": n_train, "dev": n_dev, "test": n_test})
    print(f"Dialogues: {n_dialogues}  Utterances: {n_utts}  Dim: {out[any_d][any_u].shape[0]}")

    miss_total   = miss_train   + miss_dev   + miss_test
    silent_total = silent_train + silent_dev + silent_test
    if miss_total:
        print("Missing/unreadable:", {"train": miss_train, "dev": miss_dev,
                                      "test": miss_test, "total": miss_total})
    if silent_total:
        print("Silent (zero vec):", {"train": silent_train, "dev": silent_dev,
                                     "test": silent_test, "total": silent_total})

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

    model_weight_mb = get_model_weight_mb(cfg)

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
    print(f"  wall time: {total_wall:.1f}s  (I/O + load_wav + inference)")
    print()

    if is_cuda:
        print(f"  GPU  : {torch.cuda.get_device_name(dev_idx)}")
        print(f"  Peak alloc  : {gpu_peak_mb:7.0f} MB  /  {gpu_total_mb:.0f} MB  "
              f"({gpu_peak_mb / gpu_total_mb * 100:.1f}%)")
        print(f"  Reserved    : {gpu_res_mb:7.0f} MB")
    else:
        print(f"  Device: CPU")
        print(f"  Tip   : re-run with --limit 200 --device cpu for a fast CPU baseline,")
        print(f"          then compare against GPU with the same --limit value.")

    if model_weight_mb is not None:
        print(f"  Model weights: {model_weight_mb:.0f} MB  (cache on disk, fp32)")
    else:
        print(f"  Model weights: n/a  (ModelScope cache not found at ~/.cache/modelscope)")

    if cpu_rss_mb is not None:
        print(f"  CPU RSS     : {cpu_rss_mb:7.0f} MB  (process peak working set)")
    else:
        print(f"  CPU RSS     : n/a  (install psutil for CPU memory tracking)")

    print("─" * W)


if __name__ == "__main__":
    main()