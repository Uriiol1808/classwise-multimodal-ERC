# audio/models.py
import os
import re
import io
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Tuple, Optional

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from scipy.signal import resample_poly

IEMOCAP_TRAIN_SESSIONS = [1, 2, 3, 4]
IEMOCAP_TEST_SESSIONS  = [5]
_TRANS_LINE_RE = re.compile(r"^(\S+)\s+\[\d+\.\d+-\d+\.\d+\]:\s*(.*)")
_EMO_LINE_RE   = re.compile(r"^\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+(\S+)\s+(\w+)")


@dataclass
class AudioConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_id: str = "iic/emotion2vec_plus_large"

    target_sr: int = 16000
    max_seconds: Optional[float] = None
    batch_size: int = 8
    pooling: Literal["mean"] = "mean"


# ── Encoding fix (same mojibake bug as MELD text CSVs) ────────────────────────

def _fix_meld_mojibake(raw: bytes) -> str:
    """
    Repairs the Windows-1252 mojibake present in MELD CSVs.
    After UTF-8 decode, corrupted chars land in U+0080-U+009F (C1 control range)
    which is never used in plain text. Replace only those codepoints with their
    Windows-1252 equivalents; leave all real UTF-8 characters untouched.
    """
    text = raw.decode("utf-8", errors="replace")

    def _replace_c1(m: re.Match) -> str:
        byte_val = ord(m.group(0))
        try:
            return bytes([byte_val]).decode("windows-1252")
        except (ValueError, UnicodeDecodeError):
            return m.group(0)

    fixed = re.sub(r"[\u0080-\u009f]", _replace_c1, text)
    fixed = fixed.replace("\xa0", " ")
    return fixed


def load_csv(path: str) -> pd.DataFrame:
    """Loads a MELD CSV, repairing the Windows-1252 mojibake encoding bug."""
    with open(path, "rb") as f:
        raw = f.read()
    return pd.read_csv(io.StringIO(_fix_meld_mojibake(raw)))


# Model loading
def load_audio_model(cfg: AudioConfig):
    """
    Loads emotion2vec via FunASR. Returns (processor, model, embed_dim).
    processor is always None for FunASR models (kept for API symmetry with text).
    """
    from funasr import AutoModel as FunASRAutoModel

    if cfg.model_id.startswith("emotion2vec/"):
        ms_id = "iic/" + cfg.model_id.split("/", 1)[1]
    else:
        ms_id = cfg.model_id

    model = FunASRAutoModel(model=ms_id, hub="ms")
    processor = None

    embed_dim = _infer_funasr_embed_dim(model, cfg.target_sr)
    return processor, model, embed_dim


def get_model_weight_mb(cfg: AudioConfig) -> Optional[float]:
    """
    Returns the size of the cached model weights in MB.
    FunASR models don't expose .parameters(), so we sum the safetensors/bin
    files in the ModelScope cache directory.

    Returns None if the cache cannot be located.
    """
    import glob

    # ModelScope default cache: ~/.cache/modelscope/hub/<org>/<model>
    model_name = cfg.model_id.replace("/", os.sep)
    cache_root = os.path.expanduser(os.path.join("~", ".cache", "modelscope", "hub"))
    model_dir = os.path.join(cache_root, model_name)

    if not os.path.isdir(model_dir):
        return None

    total = 0
    for ext in ("*.safetensors", "*.bin", "*.pt", "*.pth"):
        for f in glob.glob(os.path.join(model_dir, "**", ext), recursive=True):
            total += os.path.getsize(f)

    return total / 1024**2 if total > 0 else None



_TARGET_DBFS       = -20.0
_SILENCE_THRESHOLD = 1e-4
_BOUNDARY_PAD_S    = 0.05


def _to_mono(wav: np.ndarray) -> np.ndarray:
    if wav.ndim == 1:
        return wav
    return wav.mean(axis=1)


def _resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wav
    g = np.gcd(orig_sr, target_sr)
    return resample_poly(wav, target_sr // g, orig_sr // g).astype(np.float32, copy=False)


def _normalize_rms(wav: np.ndarray, target_dbfs: float = _TARGET_DBFS) -> Tuple[np.ndarray, bool]:
    rms = float(np.sqrt(np.mean(wav ** 2)))
    if rms < _SILENCE_THRESHOLD:
        return wav, True
    current_dbfs = 20.0 * np.log10(rms + 1e-9)
    gain = 10.0 ** ((target_dbfs - current_dbfs) / 20.0)
    return (wav * gain).astype(np.float32, copy=False), False


def load_wav(path: str, target_sr: int, max_seconds: Optional[float] = None, 
             normalize_audio: bool = True, boundary_pad: bool = True
             ) -> Tuple[np.ndarray, int, bool]:
    """
    Loads, resamples, truncates and normalizes a waveform.

    Order of operations:
      1. Read + mono
      2. Resample to target_sr   ← before truncation (avoids wrong sample count)
      3. Truncate to max_seconds
      4. Boundary silence padding (avoids abrupt cutoff artifacts)
      5. RMS normalize

    Returns: (wav, sr, is_silent)
    """
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    wav = _to_mono(wav)

    if sr != target_sr:
        wav = _resample(wav, sr, target_sr)
        sr = target_sr

    if max_seconds is not None and max_seconds > 0:
        wav = wav[:int(target_sr * max_seconds)]

    if wav.size == 0:
        return np.zeros((1,), dtype=np.float32), sr, True

    if boundary_pad:
        pad = np.zeros((int(_BOUNDARY_PAD_S * sr),), dtype=np.float32)
        wav = np.concatenate([pad, wav, pad])

    is_silent = False
    if normalize_audio:
        wav, is_silent = _normalize_rms(wav)

    return wav.astype(np.float32, copy=False), sr, is_silent



def _extract_funasr_embedding(res) -> np.ndarray:
    if isinstance(res, list) and len(res) > 0:
        res = res[0]
    if isinstance(res, dict):
        for k in ("embedding", "emb", "feats", "feat", "vector"):
            if k in res and res[k] is not None:
                arr = np.asarray(res[k])
                if arr.ndim == 2 and arr.shape[0] == 1:
                    arr = arr[0]
                return arr.astype(np.float32, copy=False)
    raise RuntimeError(
        f"Could not find embedding in FunASR output. "
        f"type={type(res)} keys={list(res.keys()) if isinstance(res, dict) else None}"
    )


def _write_tmp_wav(wav: np.ndarray, sr: int) -> str:
    """Writes wav to a named temp file and returns the path. Caller must delete."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    sf.write(path, wav, sr)
    return path


def _funasr_embed_one_utt(model, wav: np.ndarray, sr: int) -> np.ndarray:
    path = _write_tmp_wav(wav, sr)
    try:
        res = model.generate(path, granularity="utterance", extract_embedding=True)
        return _extract_funasr_embedding(res)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _infer_funasr_embed_dim(model, sampling_rate: int) -> int:
    wav = np.zeros((sampling_rate,), dtype=np.float32)
    return int(_funasr_embed_one_utt(model, wav, sampling_rate).shape[-1])



@torch.no_grad()
def embed_audio_batch(waves: List[np.ndarray], processor, model, device: str, 
                      sampling_rate: int, pooling: str = "mean", 
                      silent_mask: Optional[List[bool]] = None, embed_dim: Optional[int] = None
                      ) -> np.ndarray:
    """
    Embeds a batch of waveforms. Returns (B, D) float32.

    FunASR path (processor is None):
      FunASR processes each utterance sequentially internally — there is no
      true batch inference API. The loop here is explicit and honest about that.
      Silent utterances are short-circuited to zero vectors without model calls.

    Transformers path (processor is not None):
      Standard padded batch inference with attention mask mean pooling.

    silent_mask: if provided, True entries get zero vectors without model calls.
    embed_dim:   required when silent_mask contains any True entries.
    """
    B = len(waves)

    # ── FunASR path ───────────────────────────────────────────────────────────
    if processor is None:
        assert embed_dim is not None, "embed_dim required for FunASR path"
        results = np.zeros((B, embed_dim), dtype=np.float32)

        for i, wav in enumerate(waves):
            if silent_mask is not None and silent_mask[i]:
                continue  # leave as zero vector
            results[i] = _funasr_embed_one_utt(model, wav, sampling_rate)

        return results

    # ── Transformers path ─────────────────────────────────────────────────────
    inputs = processor(waves, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = model(**inputs)
    hidden = getattr(out, "last_hidden_state", None)
    if hidden is None:
        raise RuntimeError("No last_hidden_state in model output")

    if pooling != "mean":
        raise ValueError("Only pooling='mean' implemented")

    attn = inputs.get("attention_mask", None)
    if attn is None:
        emb = hidden.mean(dim=1)
    else:
        mask = attn.unsqueeze(-1).to(hidden.dtype)
        emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    return emb.detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def embed_one_audio(wav: np.ndarray, processor, model, device: str,
                    sampling_rate: int) -> np.ndarray:
    return embed_audio_batch([wav], processor, model, device, sampling_rate)[0]


# ── IEMOCAP loader ────────────────────────────────────────────────────────────

def _parse_transcription_audio(path: Path) -> list[tuple[str, str]]:
    """Returns [(utt_str_id, text), ...] in file order."""
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


def _parse_emo_eval_audio(path: Path) -> dict[str, float]:
    """Returns {utt_str_id: start_time} for temporal sorting."""
    times = {}
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = _EMO_LINE_RE.match(line.strip())
            if m:
                utt_id = m.group(3)
                if utt_id not in times:
                    times[utt_id] = float(m.group(1))
    return times


def _speaker_from_utt_id(utt_id: str) -> str:
    part = utt_id.rsplit("_", 1)[-1]
    return part[0] if part and part[0] in ("M", "F") else "U"


def load_iemocap_audio_dataframes(
    iemocap_root: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, str]]:
    """
    Parse IEMOCAP filesystem into (train_df, empty_dev_df, test_df, id_map).

    DataFrame columns:
      Dialogue_ID  (int)   — sequential integer
      Utterance_ID (int)   — 0-based position within dialogue, sorted by time
      Audio_Path   (str)   — absolute path to per-utterance WAV
      Speaker      (str)   — 'M' or 'F'
      Utt_Str_ID   (str)   — e.g. 'Ses01F_impro01_F000'

    Sessions 1-4 → train, Session 5 → test. No dev split.
    ALL utterances included; emotion filtering happens at assembly step.
    """
    root = Path(iemocap_root)
    rows: list[dict] = []
    id_map: dict[int, str] = {}
    dialogue_counter = 0

    for split_sessions in [IEMOCAP_TRAIN_SESSIONS, IEMOCAP_TEST_SESSIONS]:
        for sess in split_sessions:
            trans_dir    = root / f"Session{sess}" / "dialog" / "transcriptions"
            emo_eval_dir = root / f"Session{sess}" / "dialog" / "EmoEvaluation"
            wav_root     = root / f"Session{sess}" / "sentences" / "wav"

            for trans_file in sorted(
                f for f in trans_dir.glob("*.txt") if not f.name.startswith("._")
            ):
                dialog_name = trans_file.stem
                emo_file    = emo_eval_dir / f"{dialog_name}.txt"
                start_times = _parse_emo_eval_audio(emo_file) if emo_file.exists() else {}

                entries = _parse_transcription_audio(trans_file)
                if not entries:
                    continue

                entries.sort(key=lambda x: start_times.get(x[0], float("inf")))

                d_int = dialogue_counter
                id_map[d_int] = dialog_name
                dialogue_counter += 1

                for u_idx, (utt_str_id, _) in enumerate(entries):
                    wav_path = wav_root / dialog_name / f"{utt_str_id}.wav"
                    rows.append({
                        "Dialogue_ID":  d_int,
                        "Utterance_ID": u_idx,
                        "Audio_Path":   str(wav_path),
                        "Speaker":      _speaker_from_utt_id(utt_str_id),
                        "Utt_Str_ID":   utt_str_id,
                    })

    df_all = pd.DataFrame(rows)
    train_mask = df_all["Utt_Str_ID"].str.startswith(
        tuple(f"Ses0{s}" for s in IEMOCAP_TRAIN_SESSIONS)
    )
    train_df     = df_all[train_mask].reset_index(drop=True)
    test_df      = df_all[~train_mask].reset_index(drop=True)
    empty_dev_df = pd.DataFrame(columns=df_all.columns)

    print(f"IEMOCAP loaded: {len(train_df)} train utterances, "
          f"{len(test_df)} test utterances across {dialogue_counter} dialogues.")
    return train_df, empty_dev_df, test_df, id_map