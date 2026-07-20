"""
extract_degraded_audio.py

Applies waveform-level degradation BEFORE emotion2vec embedding for the
test set only, producing a standalone audio pkl that assemble_degraded_pkl.py
can patch into the combined pkl.

Supports both IEMOCAP and MELD.  For MELD, train/dev CSVs are required
(read-only) to compute the correct test-split dialogue ID offset — which must
match the remapping used when the clean pkl was built.

Degradations (teleassistance setting):
  noise       — AWGN at specified SNR (models mic noise, home ambient sound)
  packet_loss — VoIP 20ms packet silence (models network dropouts)

Usage
-----
# IEMOCAP — noise
python degrade_audio.py --dataset iemocap \
--iemocap_root /media/ssd2/oriol/IEMOCAP/IEMOCAP_full_release \
--out /media/ssd2/oriol/IEMOCAP/IEMOCAP_full_release/embeddings/degraded/audio/snr/iemocap_audio_noise_snr20.pkl \
--degradation noise --snr_db 20

python degrade_audio.py --dataset iemocap \
    --iemocap_root /media/ssd2/oriol/IEMOCAP/IEMOCAP_full_release \
    --out /media/ssd2/oriol/IEMOCAP/IEMOCAP_full_release/embeddings/degraded/audio/snr/iemocap_audio_noise_snr10.pkl \
    --degradation noise --snr_db 10

python degrade_audio.py --dataset iemocap \
    --iemocap_root /media/ssd2/oriol/IEMOCAP/IEMOCAP_full_release \
    --out /media/ssd2/oriol/IEMOCAP/IEMOCAP_full_release/embeddings/degraded/audio/snr/iemocap_audio_noise_snr5.pkl \
    --degradation noise --snr_db 5

python degrade_audio.py --dataset iemocap \
    --iemocap_root /media/ssd2/oriol/IEMOCAP/IEMOCAP_full_release \
    --out /media/ssd2/oriol/IEMOCAP/IEMOCAP_full_release/embeddings/degraded/audio/snr/iemocap_audio_noise_snr0.pkl \
    --degradation noise --snr_db 0

python degrade_audio.py --dataset iemocap \
    --iemocap_root /media/ssd2/oriol/IEMOCAP/IEMOCAP_full_release \
    --out /media/ssd2/oriol/IEMOCAP/IEMOCAP_full_release/embeddings/degraded/audio/packet_loss/iemocap_audio_packet_loss01.pkl \
    --degradation packet_loss --rate 0.1

python degrade_audio.py --dataset iemocap \
    --iemocap_root /media/ssd2/oriol/IEMOCAP/IEMOCAP_full_release \
    --out /media/ssd2/oriol/IEMOCAP/IEMOCAP_full_release/embeddings/degraded/audio/packet_loss/iemocap_audio_packet_loss02.pkl \
    --degradation packet_loss --rate 0.2

python degrade_audio.py --dataset iemocap \
    --iemocap_root /media/ssd2/oriol/IEMOCAP/IEMOCAP_full_release \
    --out /media/ssd2/oriol/IEMOCAP/IEMOCAP_full_release/embeddings/degraded/audio/packet_loss/iemocap_audio_packet_loss03.pkl \
    --degradation packet_loss --rate 0.3

python degrade_audio.py --dataset iemocap \
    --iemocap_root /media/ssd2/oriol/IEMOCAP/IEMOCAP_full_release \
    --out /media/ssd2/oriol/IEMOCAP/IEMOCAP_full_release/embeddings/degraded/audio/packet_loss/iemocap_audio_packet_loss05.pkl \
    --degradation packet_loss --rate 0.5

# MELD — packet loss 30%
python degrade_audio.py --dataset meld \
    --train_csv /media/ssd2/oriol/MELD/train_sent_emo.csv \
    --dev_csv   /media/ssd2/oriol/MELD/dev_sent_emo.csv \
    --test_csv  /media/ssd2/oriol/MELD/test_sent_emo.csv \
    --test_audio_dir /media/ssd2/oriol/MELD/test/wav16k \
    --out /media/ssd2/oriol/MELD/embeddings/degraded/audio/meld_audio_noise_snr5.pkl \
    --degradation noise --snr_db 5

python degrade_audio.py --dataset meld \
    --train_csv /media/ssd2/oriol/MELD/train_sent_emo.csv \
    --dev_csv   /media/ssd2/oriol/MELD/dev_sent_emo.csv \
    --test_csv  /media/ssd2/oriol/MELD/test_sent_emo.csv \
    --test_audio_dir /media/ssd2/oriol/MELD/test/wav16k \
    --out /media/ssd2/oriol/MELD/embeddings/degraded/meld_audio_packet_loss01.pkl \
    --degradation packet_loss --rate 0.1

python degrade_audio.py --dataset meld \
    --train_csv /media/ssd2/oriol/MELD/train_sent_emo.csv \
    --dev_csv   /media/ssd2/oriol/MELD/dev_sent_emo.csv \
    --test_csv  /media/ssd2/oriol/MELD/test_sent_emo.csv \
    --test_audio_dir /media/ssd2/oriol/MELD/test/wav16k \
    --out /media/ssd2/oriol/MELD/embeddings/degraded/meld_audio_packet_loss02.pkl \
    --degradation packet_loss --rate 0.2
    
python degrade_audio.py --dataset meld \
    --train_csv /media/ssd2/oriol/MELD/train_sent_emo.csv \
    --dev_csv   /media/ssd2/oriol/MELD/dev_sent_emo.csv \
    --test_csv  /media/ssd2/oriol/MELD/test_sent_emo.csv \
    --test_audio_dir /media/ssd2/oriol/MELD/test/wav16k \
    --out /media/ssd2/oriol/MELD/embeddings/degraded/meld_audio_packet_loss03.pkl \
    --degradation packet_loss --rate 0.3

python degrade_audio.py --dataset meld \
    --train_csv /media/ssd2/oriol/MELD/train_sent_emo.csv \
    --dev_csv   /media/ssd2/oriol/MELD/dev_sent_emo.csv \
    --test_csv  /media/ssd2/oriol/MELD/test_sent_emo.csv \
    --test_audio_dir /media/ssd2/oriol/MELD/test/wav16k \
    --out /media/ssd2/oriol/MELD/embeddings/degraded/meld_audio_packet_loss05.pkl \
    --degradation packet_loss --rate 0.5

    
Suggested sweeps
----------------
  noise       : --snr_db   20 10 5 0
  packet_loss : --rate  0.1 0.2 0.3 0.5
"""

import os
import argparse
import pickle
import numpy as np
from tqdm import tqdm

from .models import (
    AudioConfig,
    load_audio_model,
    load_csv,
    load_iemocap_audio_dataframes,
    load_wav,
    _funasr_embed_one_utt,
)


# ── waveform-level degradations ───────────────────────────────────────────────

def degrade_noise(waveform: np.ndarray, snr_db: float) -> np.ndarray:
    """AWGN at given SNR (dB). snr_db=0 → noise power = signal power."""
    rms       = float(np.sqrt(np.mean(waveform ** 2))) + 1e-9
    noise_rms = rms / (10.0 ** (snr_db / 20.0))
    return waveform + np.random.randn(len(waveform)).astype(np.float32) * noise_rms


def degrade_packet_loss(waveform: np.ndarray, sr: int,
                        rate: float, packet_ms: float = 20.0) -> np.ndarray:
    """Silence `rate` fraction of 20 ms VoIP packets at random."""
    packet_len = max(1, int(sr * packet_ms / 1000))
    out        = waveform.copy()
    for i in range(len(waveform) // packet_len):
        if np.random.rand() < rate:
            out[i * packet_len:(i + 1) * packet_len] = 0.0
    return out


# ── dataset helpers ───────────────────────────────────────────────────────────

def _load_meld_test(train_csv, dev_csv, test_csv, test_audio_dir, audio_ext):
    """
    Returns (test_df_with_paths, test_map, id_map=None).
    test_map: {original_dialogue_id → remapped_id} using the same offset
    formula as main_audio.py so keys match the clean combined pkl.
    """
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
    test_df["Dialogue_ID"] = test_df["Dialogue_ID"].astype(int)
    test_df["Utterance_ID"] = test_df["Utterance_ID"].astype(int)

    if "Audio_Path" not in test_df.columns:
        test_df["Audio_Path"] = test_df.apply(
            lambda r: os.path.join(
                test_audio_dir,
                f"dia{int(r['Dialogue_ID'])}_utt{int(r['Utterance_ID'])}{audio_ext}"
            ), axis=1
        )

    return test_df, test_map, None   # no id_map for MELD


def _load_iemocap_test(iemocap_root):
    """
    Returns (test_df_with_paths, test_map, id_map).
    test_map keeps original integer IDs; id_map resolves them to dialogue names.
    """
    _, _, test_df, id_map = load_iemocap_audio_dataframes(iemocap_root)
    test_ids = sorted(test_df["Dialogue_ID"].astype(int).unique())
    test_map = {d: d for d in test_ids}   # keep original — id_map bridges to names
    return test_df, test_map, id_map


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--dataset", required=True, choices=["iemocap", "meld"])

    # IEMOCAP
    ap.add_argument("--iemocap_root", default=None)

    # MELD
    ap.add_argument("--train_csv",       default=None)
    ap.add_argument("--dev_csv",         default=None)
    ap.add_argument("--test_csv",        default=None)
    ap.add_argument("--test_audio_dir",  default=None)
    ap.add_argument("--audio_ext",       default=".wav")

    ap.add_argument("--out",         required=True)
    ap.add_argument("--model_id",    default="iic/emotion2vec_plus_large")
    ap.add_argument("--device",      default="cuda")
    ap.add_argument("--target_sr",   type=int,   default=16000)
    ap.add_argument("--max_seconds", type=float, default=10.0)

    ap.add_argument("--degradation", required=True, choices=["noise", "packet_loss"])
    ap.add_argument("--snr_db", type=float, default=10.0,
                    help="[noise] SNR in dB. Sweep: 20 10 5 0")
    ap.add_argument("--rate",   type=float, default=0.3,
                    help="[packet_loss] Fraction of 20ms packets silenced. Sweep: 0.1 0.2 0.3 0.5")
    args = ap.parse_args()

    # ── validate args ──────────────────────────────────────────────────────────
    if args.dataset == "iemocap" and not args.iemocap_root:
        ap.error("--dataset iemocap requires --iemocap_root")
    if args.dataset == "meld" and not all(
        [args.train_csv, args.dev_csv, args.test_csv, args.test_audio_dir]
    ):
        ap.error("--dataset meld requires --train_csv, --dev_csv, --test_csv, --test_audio_dir")

    tag = (f"noise_snr{args.snr_db}dB" if args.degradation == "noise"
           else f"pktloss{int(args.rate * 100)}pct")
    print(f"Dataset: {args.dataset}  |  Degradation: {tag}")

    # ── load data ──────────────────────────────────────────────────────────────
    if args.dataset == "iemocap":
        test_df, test_map, id_map = _load_iemocap_test(args.iemocap_root)
    else:
        test_df, test_map, id_map = _load_meld_test(
            args.train_csv, args.dev_csv, args.test_csv,
            args.test_audio_dir, args.audio_ext
        )
    print(f"Test utterances: {len(test_df)}")

    # ── load model ─────────────────────────────────────────────────────────────
    cfg = AudioConfig(
        device=args.device,
        model_id=args.model_id,
        target_sr=args.target_sr,
        max_seconds=args.max_seconds if args.max_seconds > 0 else None,
    )
    _, model, embed_dim = load_audio_model(cfg)
    print(f"embed_dim: {embed_dim}")

    # ── extract ────────────────────────────────────────────────────────────────
    out: dict = {}
    n_ok = n_missing = 0

    for _, row in tqdm(test_df.iterrows(), total=len(test_df),
                       desc=f"Degraded audio ({tag})"):
        d_orig = int(row["Dialogue_ID"])
        u      = int(row["Utterance_ID"])
        d_new  = test_map[d_orig]
        wav_path = str(row["Audio_Path"])

        if not os.path.exists(wav_path):
            out.setdefault(d_new, {})[u] = np.zeros(embed_dim, dtype=np.float32)
            n_missing += 1
            continue

        try:
            waveform, sr, is_silent = load_wav(
                wav_path, target_sr=cfg.target_sr, max_seconds=cfg.max_seconds
            )
        except Exception as e:
            print(f"  [WARN] {wav_path}: {e}")
            out.setdefault(d_new, {})[u] = np.zeros(embed_dim, dtype=np.float32)
            n_missing += 1
            continue

        if is_silent:
            out.setdefault(d_new, {})[u] = np.zeros(embed_dim, dtype=np.float32)
            n_ok += 1
            continue

        if args.degradation == "noise":
            waveform = degrade_noise(waveform, snr_db=args.snr_db)
        else:
            waveform = degrade_packet_loss(waveform, sr, rate=args.rate)

        out.setdefault(d_new, {})[u] = _funasr_embed_one_utt(model, waveform, sr)
        n_ok += 1

    out = {d: dict(sorted(u_m.items())) for d, u_m in sorted(out.items())}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    # id_map is None for MELD (not needed), saved for IEMOCAP
    id_map_path = args.out.replace(".pkl", "_id_map.pkl")
    with open(id_map_path, "wb") as f:
        pickle.dump(id_map, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved {sum(len(v) for v in out.values())} utterances → {args.out}")
    print(f"OK: {n_ok}  Missing/failed: {n_missing}")


if __name__ == "__main__":
    main()