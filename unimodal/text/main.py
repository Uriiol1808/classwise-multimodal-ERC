import os
import time
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from .models import (
    TextConfig,
    load_csv,
    load_iemocap_dataframes,
    add_iemocap_sentiment,
    iemocap_valence_report,
    load_text_model,
    embed_text_batch,
    build_context_index,
    build_contextual_text,
)


def add_split_text(out_nested: dict, df: pd.DataFrame, d_remap: dict[int, int], tokenizer,
                   model, cfg: TextConfig, desc: str) -> tuple[int, float, float]:
    """
    Extracts text embeddings for a split and writes them into out_nested:
        out_nested[new_dialogue_id][utterance_id] = embedding (np.ndarray)

    If cfg.context_window > 0, each utterance is enriched with the previous
    `context_window` utterances from the same dialogue before encoding.
    """
    required = ["Dialogue_ID", "Utterance_ID", "Utterance"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"{desc}: missing column '{c}'. Columns: {list(df.columns)}")

    df = df.copy()
    df["Dialogue_ID"] = df["Dialogue_ID"].astype(int)
    df["Utterance_ID"] = df["Utterance_ID"].astype(int)

    use_context = cfg.context_window > 0

    # CHANGED: pass sentiment_col so context_index captures sentiment per utterance
    context_index = build_context_index(
        df,
        cfg.speaker_col if cfg.include_speaker else None,
        cfg.sentiment_col if cfg.use_sentiment_signal else None,
    ) if use_context else {}

    batch_meta: list[tuple[int, int]] = []
    batch_texts: list[str] = []
    inference_times: list[float] = []
    inference_utts: list[int] = []

    def flush():
        if not batch_texts:
            return
        t0 = time.perf_counter()
        E = embed_text_batch(
            batch_texts, tokenizer, model,
            cfg.device, cfg.max_length, pooling=cfg.pooling,
        )
        if cfg.device.startswith("cuda"):
            torch.cuda.synchronize()
        inference_times.append(time.perf_counter() - t0)
        inference_utts.append(len(batch_texts))

        for (dd, uu), e in zip(batch_meta, E):
            out_nested.setdefault(dd, {})[uu] = np.asarray(e)
        batch_meta.clear()
        batch_texts.clear()

    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        d_orig = int(row["Dialogue_ID"])
        u = int(row["Utterance_ID"])
        d_new = d_remap[d_orig]

        if use_context:
            # CHANGED: forward task_prefix and use_sentiment_signal
            txt = build_contextual_text(
                d_id=d_orig, u_id=u,
                context_index=context_index,
                context_window=cfg.context_window,
                context_sep=cfg.context_sep,
                include_speaker=cfg.include_speaker,
                task_prefix=cfg.task_prefix,
                use_sentiment_signal=cfg.use_sentiment_signal,
            )
        else:
            raw = row["Utterance"]
            txt = "" if pd.isna(raw) else str(raw)
            if cfg.task_prefix:
                txt = f"{cfg.task_prefix}{txt}"

        batch_meta.append((d_new, u))
        batch_texts.append(txt)

        if len(batch_texts) >= cfg.batch_size:
            flush()

    flush()

    total_s = sum(inference_times)
    total_u = sum(inference_utts)
    avg_ms = (total_s / total_u * 1000) if total_u > 0 else 0.0
    return len(df), total_s, avg_ms


def main():
    ap = argparse.ArgumentParser()

    # ── dataset source ────────────────────────────────────────────────────────
    ap.add_argument("--dataset", choices=["meld", "iemocap"], default="meld",
                    help="Which dataset to extract from.")
    # MELD args (ignored when --dataset iemocap)
    ap.add_argument("--train_csv", default=None)
    ap.add_argument("--dev_csv",   default=None)
    ap.add_argument("--test_csv",  default=None)
    # IEMOCAP args (ignored when --dataset meld)
    ap.add_argument("--iemocap_root", default=None,
                    help="Path to IEMOCAP_full_release directory.")
    ap.add_argument("--iemocap_va_neg_thresh", type=float, default=2.0,
                    help="Valence <= this -> 'negative' Sentiment token for IEMOCAP context "
                         "(mirrors MELD's Sentiment column). Valence is on EmoEvaluation's 1-5 scale.")
    ap.add_argument("--iemocap_va_pos_thresh", type=float, default=3.0,
                    help="Valence >= this -> 'positive' Sentiment token for IEMOCAP context.")

    ap.add_argument("--out", required=True)

    ap.add_argument("--model_id", default="sentence-transformers/all-roberta-large-v1")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--pooling", choices=["cls", "mean", "cls+mean"], default="mean")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--context_window", type=int, default=2)
    ap.add_argument("--no_speaker", action="store_true")
    ap.add_argument("--speaker_col", default="Speaker")
    ap.add_argument("--context_sep", default=" [SEP] ")

    # NEW args
    ap.add_argument("--task_prefix", default="", help="Leave empty for all-roberta/all-mpnet. E5 requires 'query: Classify the emotion in this conversation utterance: '")
    ap.add_argument("--no_sentiment", action="store_true", help="Disable appending sentiment token to target utterance.")
    ap.add_argument("--sentiment_col", default="Sentiment")

    ap.add_argument("--limit", type=int, default=0,
                    help="Rows per split. 0 = all. Use --limit 100 --device cpu for a fast "
                         "CPU timing baseline without running the full dataset.")

    args = ap.parse_args()

    limit_rows = None if not args.limit or args.limit <= 0 else int(args.limit)

    cfg = TextConfig(
        device=args.device,
        model_id=args.model_id,
        max_length=args.max_length,
        batch_size=args.batch_size,
        pooling=args.pooling,
        context_window=args.context_window,
        context_sep=args.context_sep,
        include_speaker=not args.no_speaker,
        speaker_col=args.speaker_col,
        task_prefix=args.task_prefix,
        use_sentiment_signal=not args.no_sentiment,
        sentiment_col=args.sentiment_col,
    )

    print("Configuration:")
    print(f"  dataset             : {args.dataset}")
    print(f"  model_id            : {cfg.model_id}")
    print(f"  pooling             : {cfg.pooling}")
    print(f"  context_window      : {cfg.context_window}")
    print(f"  context_sep         : {repr(cfg.context_sep)}")
    print(f"  include_speaker     : {cfg.include_speaker}")
    print(f"  task_prefix         : {repr(cfg.task_prefix)}")
    print(f"  use_sentiment_signal: {cfg.use_sentiment_signal}")
    print(f"  max_length          : {cfg.max_length}")
    print(f"  batch_size          : {cfg.batch_size}")
    print(f"  device              : {cfg.device}")
    print()

    tokenizer, model, embed_dim = load_text_model(cfg)
    print(f"  embed_dim           : {embed_dim}  (hidden={model.config.hidden_size}, pooling={cfg.pooling})")
    print()

    # ── load data ─────────────────────────────────────────────────────────────
    id_map = None  # only populated for IEMOCAP

    if args.dataset == "meld":
        if not (args.train_csv and args.dev_csv and args.test_csv):
            ap.error("--dataset meld requires --train_csv, --dev_csv, --test_csv")
        print(f"  encoding fix        : windows-1252 mojibake → UTF-8 (load_csv)")
        train_df = load_csv(args.train_csv)
        dev_df   = load_csv(args.dev_csv)
        test_df  = load_csv(args.test_csv)
    else:  # iemocap
        if not args.iemocap_root:
            ap.error("--dataset iemocap requires --iemocap_root")
        train_df, dev_df, test_df, id_map = load_iemocap_dataframes(args.iemocap_root)

        # Derive a MELD-style Sentiment column (positive/negative/neutral) from
        # the per-utterance dimensional valence rating, so use_sentiment_signal
        # works the same way it does for MELD.
        train_df = add_iemocap_sentiment(train_df, args.iemocap_va_neg_thresh, args.iemocap_va_pos_thresh)
        test_df  = add_iemocap_sentiment(test_df,  args.iemocap_va_neg_thresh, args.iemocap_va_pos_thresh)

        if cfg.use_sentiment_signal:
            print(f"  IEMOCAP Sentiment thresholds: valence <= {args.iemocap_va_neg_thresh} -> negative, "
                  f">= {args.iemocap_va_pos_thresh} -> positive, else neutral")
            print("  Sentiment distribution (train):", train_df["Sentiment"].value_counts().to_dict())
            print("  Sentiment distribution (test) :", test_df["Sentiment"].value_counts().to_dict())
            print("  Valence by EmoCategory (train):")
            print(iemocap_valence_report(train_df).to_string())
            print()

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

    out: dict = {}
    t_start = time.perf_counter()
    n_train, s_train, ms_train = add_split_text(out, train_df, train_map, tokenizer, model, cfg, desc="Extracting text (train)")
    n_dev,   s_dev,   ms_dev   = add_split_text(out, dev_df,   dev_map,   tokenizer, model, cfg, desc="Extracting text (dev)")
    n_test,  s_test,  ms_test  = add_split_text(out, test_df,  test_map,  tokenizer, model, cfg, desc="Extracting text (test)")
    total_wall = time.perf_counter() - t_start

    out = {
        d: dict(sorted(u_map.items()))
        for d, u_map in sorted(out.items())
    }

    with open(args.out, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    # For IEMOCAP: save the int→dialogue_name mapping next to the embeddings
    # so the assembly script can reconstruct which row belongs to which dialog.
    if id_map is not None:
        id_map_path = args.out.replace(".pkl", "_id_map.pkl")
        with open(id_map_path, "wb") as f:
            pickle.dump(id_map, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved id_map → {id_map_path}")

    n_dialogues = len(out)
    n_utts      = sum(len(u_map) for u_map in out.values())
    any_d       = next(iter(out))
    any_u       = next(iter(out[any_d]))
    actual_dim  = out[any_d][any_u].shape[0]

    print("\nSaved:", args.out)
    print("Processed rows:", {"train": n_train, "dev": n_dev, "test": n_test})
    print(f"Dialogues: {n_dialogues}  Utterances: {n_utts}  Dim: {actual_dim}")
    print(f"Example access: out[{any_d}][{any_u}].shape = {out[any_d][any_u].shape}")

    total_inf  = s_train + s_dev + s_test
    total_utts = n_train + n_dev + n_test
    avg_ms_all = (total_inf / total_utts * 1000) if total_utts > 0 else 0.0

    # ── Memory snapshot ───────────────────────────────────────────────────────
    is_cuda = cfg.device.startswith("cuda") and torch.cuda.is_available()

    if is_cuda:
        dev_idx      = torch.cuda.current_device()
        gpu_peak_mb  = torch.cuda.max_memory_allocated(dev_idx) / 1024**2
        gpu_res_mb   = torch.cuda.memory_reserved(dev_idx)      / 1024**2
        gpu_total_mb = torch.cuda.get_device_properties(dev_idx).total_memory / 1024**2

    # CPU RSS: current process peak working set.
    # psutil is optional — degrades gracefully if not installed.
    cpu_rss_mb = None
    try:
        import psutil, os
        cpu_rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    except ImportError:
        pass   # install with: pip install psutil

    # ── Report ────────────────────────────────────────────────────────────────
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
    print(f"  wall time: {total_wall:.1f}s  (I/O + tokenization + inference)")
    print()

    if is_cuda:
        print(f"  GPU  : {torch.cuda.get_device_name(dev_idx)}")
        print(f"  Peak alloc  : {gpu_peak_mb:7.0f} MB  /  {gpu_total_mb:.0f} MB  "
              f"({gpu_peak_mb / gpu_total_mb * 100:.1f}%)")
        print(f"  Reserved    : {gpu_res_mb:7.0f} MB")
    else:
        print(f"  Device: CPU")
        print(f"  Note  : GPU memory tracking unavailable on CPU.")
        print(f"  Tip   : re-run with --limit 100 on CPU to get a fast timing baseline,")
        print(f"          then compare against GPU with the same --limit value.")

    if cpu_rss_mb is not None:
        model_params = sum(p.numel() for p in model.parameters())
        gpu_model_mb = model_params * 2 / 1024**2   # fp16 = 2 bytes
        cpu_model_mb = model_params * 4 / 1024**2   # fp32 = 4 bytes
        print(f"  Model weights (fp16): {gpu_model_mb:.0f} MB  (GPU)")
        print(f"  Model weights (fp32): {cpu_model_mb:.0f} MB  (CPU equivalent)")
        print(f"  CPU RSS     : {cpu_rss_mb:7.0f} MB  (process peak working set)")
    else:
        print(f"  CPU RSS     : n/a  (install psutil for CPU memory tracking)")

    print("─" * W)


if __name__ == "__main__":
    main()