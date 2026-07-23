import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


@dataclass
class TextConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    model_id: str = "sentence-transformers/all-roberta-large-v1"

    # 128 is plenty — MELD utterances avg ~8 words, max context ~25 words.
    # 256 wastes 2–4× compute. Increase only if context_window > 4.
    max_length: int = 128

    batch_size: int = 16

    pooling: Literal["cls", "mean", "cls+mean"] = "mean"

    context_window: int = 2

    context_sep: str = " [SEP] "

    include_speaker: bool = True

    speaker_col: str = "Speaker"

    task_prefix: str = ""

    # Use the Sentiment column (positive/negative/neutral) as a soft signal:
    # appends a sentiment token to the target utterance in the context string.
    # e.g. "Rachel: I can't believe it [negative]"
    use_sentiment_signal: bool = True
    sentiment_col: str = "Sentiment"


def _fix_meld_mojibake(raw: bytes) -> str:
    """
    Repairs the mixed-encoding bug present in MELD CSVs.

    The bug: some CSVs were originally Windows-1252 and got partially
    "converted" to UTF-8 by prepending \\xc2 to each high byte, turning e.g.
    the apostrophe 0x92 into the two-byte sequence \\xc2\\x92. After a correct
    UTF-8 decode these land in the Unicode C1 control range U+0080-U+009F,
    which is never used in plain text. Real UTF-8 characters (e.g. U+2019 ')
    decode normally and must not be touched.

    Fix: decode as UTF-8, then replace only codepoints in U+0080-U+009F with
    their Windows-1252 equivalents. This is the exact range the mojibake chars
    occupy after decoding, and nothing legitimate lives there. Handles mixed
    files where some rows were correctly UTF-8 and others still carry mojibake.
    """
    text = raw.decode("utf-8", errors="replace")

    def _replace_c1(m: re.Match) -> str:
        byte_val = ord(m.group(0))          # U+0080-U+009F maps 1:1 to byte 0x80-0x9F
        try:
            return bytes([byte_val]).decode("windows-1252")
        except (ValueError, UnicodeDecodeError):
            return m.group(0)               # leave unchanged if not a valid win-1252 char

    fixed = re.sub(r"[\u0080-\u009f]", _replace_c1, text)
    fixed = fixed.replace("\xa0", " ")      # non-breaking space → regular space
    return fixed


def load_csv(path: str) -> pd.DataFrame:
    """
    Loads a MELD CSV, repairing the Windows-1252 mojibake encoding bug.
    See _fix_meld_mojibake for full details.
    """
    with open(path, "rb") as f:
        raw = f.read()

    return pd.read_csv(io.StringIO(_fix_meld_mojibake(raw)))


def load_text_model(cfg: TextConfig):
    """
    Loads tokenizer and model. Returns (tokenizer, model, embed_dim).

    For sentence-transformers, AutoModel gives us the backbone trained end-to-end
    with the sentence objective — we do NOT strip a head, the model IS the encoder.
    Mean pooling over last_hidden_state is the correct extraction method per SBERT.
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)

    use_fp16 = cfg.device.startswith("cuda") and torch.cuda.is_available()
    dtype = torch.float16 if use_fp16 else torch.float32

    model = AutoModel.from_pretrained(cfg.model_id, torch_dtype=dtype).to(cfg.device).eval()

    if cfg.device.startswith("cuda"):
        torch.cuda.empty_cache()

    hidden = int(model.config.hidden_size)
    embed_dim = hidden * 2 if cfg.pooling == "cls+mean" else hidden

    return tokenizer, model, embed_dim


@torch.no_grad()
def embed_text_batch(texts: List[str], tokenizer, model, device: str,
                     max_length: int, pooling: str = "mean") -> np.ndarray:
    """
    Embeds a batch of texts. Returns (B, H) or (B, 2H) float32.

    For sentence-transformers, mean pooling over the attention mask is the
    canonical extraction strategy (same as sentence-transformers library internals).
    """
    enc = tokenizer(texts, padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    last = out.last_hidden_state  # (B, T, H)

    if pooling == "cls":
        emb = last[:, 0, :]

    elif pooling == "mean":
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    elif pooling == "cls+mean":
        cls_emb = last[:, 0, :]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        mean_emb = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        emb = torch.cat([cls_emb, mean_emb], dim=-1)

    else:
        raise ValueError(f"pooling must be 'cls', 'mean', or 'cls+mean'. Got: {pooling!r}")

    return emb.detach().cpu().float().numpy().astype(np.float32)


def build_context_index(df: pd.DataFrame, speaker_col: Optional[str],
                        sentiment_col: Optional[str]) -> dict:
    """
    Builds O(1) context index. Captures sentiment per utterance so
    build_contextual_text can optionally append it as a soft signal.

    context_index[dialogue_id] = [
        (utt_id, text, speaker_or_None, sentiment_or_None),
        ...   ordered by Utterance_ID
    ]
    """
    index: dict = {}
    df = df.copy()
    df["Dialogue_ID"] = df["Dialogue_ID"].astype(int)
    df["Utterance_ID"] = df["Utterance_ID"].astype(int)

    has_speaker = speaker_col is not None and speaker_col in df.columns
    has_sent = sentiment_col is not None and sentiment_col in df.columns

    for d_id, grp in df.groupby("Dialogue_ID"):
        grp = grp.sort_values("Utterance_ID")
        utts = []
        for _, row in grp.iterrows():
            txt = "" if pd.isna(row["Utterance"]) else str(row["Utterance"])
            spk = str(row[speaker_col]) if has_speaker else None
            snt = str(row[sentiment_col]) if has_sent else None
            utts.append((int(row["Utterance_ID"]), txt, spk, snt))
        index[int(d_id)] = utts

    return index



# ── IEMOCAP loader ────────────────────────────────────────────────────────────

_TRANS_LINE_RE = re.compile(r"^(\S+)\s+\[\d+\.\d+-\d+\.\d+\]:\s*(.*)")
_EMO_LINE_RE   = re.compile(
    r"^\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+(\S+)\s+(\w+)\s+"
    r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]"
)

IEMOCAP_TRAIN_SESSIONS = [1, 2, 3, 4]
IEMOCAP_TEST_SESSIONS  = [5]


def _parse_transcription(path: Path) -> list[tuple[str, str]]:
    """
    Parse a transcription file. Returns [(utt_str_id, text), ...] in file order
    (which is already temporal order).

    Format: Ses01F_impro01_F000 [006.29-008.24]: Excuse me.
    """
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


def _parse_emo_eval(path: Path) -> dict[str, dict]:
    """
    Parse an EmoEvaluation file. Returns:
        {utt_str_id: {"start": float, "emotion": str, "valence": float}}

    "start" lets us sort transcription utterances by time when the file
    order is ambiguous. "valence" is the line-level dimensional V rating
    (1-5 scale), used to derive a MELD-style Sentiment token. "emotion" is
    the raw 3-4 letter categorical code (e.g. "neu", "hap", "xxx"), kept for
    diagnostics only. Only the first (majority-vote) line per utterance is
    used.
    """
    info = {}
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = _EMO_LINE_RE.match(line.strip())
            if m:
                utt_id = m.group(3)
                if utt_id not in info:
                    info[utt_id] = {
                        "start": float(m.group(1)),
                        "emotion": m.group(4),
                        "valence": float(m.group(5)),
                    }
    return info


def _speaker_from_utt_id(utt_id: str) -> str:
    """Ses01F_impro01_F000 → 'F',  Ses01F_impro01_M003 → 'M'"""
    part = utt_id.rsplit("_", 1)[-1]  # e.g. 'F000'
    return part[0] if part and part[0] in ("M", "F") else "U"


def _dialog_name_from_utt_id(utt_id: str) -> str:
    """Ses01F_impro01_F000 → Ses01F_impro01"""
    return "_".join(utt_id.split("_")[:-1])


def load_iemocap_dataframes(
    iemocap_root: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, str]]:
    """
    Parse IEMOCAP filesystem into (train_df, empty_dev_df, test_df, id_map).

    id_map: {new_int_dialogue_id → original_dialogue_name_str}
      saved alongside the embeddings pkl so downstream assembly scripts can
      recover which embedding row belongs to which utterance.

    DataFrame columns:
      Dialogue_ID  (int)   — sequential integer, unique across train+test
      Utterance_ID (int)   — 0-based position within dialogue, sorted by time
      Utterance    (str)   — raw transcript text
      Speaker      (str)   — 'M' or 'F'
      Utt_Str_ID   (str)   — original ID e.g. 'Ses01F_impro01_F000'
                             (ignored by add_split_text, used by assembly)

    IEMOCAP has no dev split → dev_df is an empty DataFrame with the same columns.
    Sessions 1-4 → train, Session 5 → test.
    ALL utterances are included (even 'xxx' labels); emotion filtering happens
    at the final pickle assembly step.
    """
    root = Path(iemocap_root)
    rows: list[dict] = []
    id_map: dict[int, str] = {}
    dialogue_counter = 0

    for split_sessions in [IEMOCAP_TRAIN_SESSIONS, IEMOCAP_TEST_SESSIONS]:
        for sess in split_sessions:
            trans_dir    = root / f"Session{sess}" / "dialog" / "transcriptions"
            emo_eval_dir = root / f"Session{sess}" / "dialog" / "EmoEvaluation"

            for trans_file in sorted(f for f in trans_dir.glob("*.txt") if not f.name.startswith("._")):
                dialog_name = trans_file.stem

                # Get start times for sorting (transcription order is usually
                # already temporal, but EmoEval is the authoritative source)
                emo_file = emo_eval_dir / f"{dialog_name}.txt"
                emo_info = _parse_emo_eval(emo_file) if emo_file.exists() else {}

                entries = _parse_transcription(trans_file)
                if not entries:
                    continue

                # Sort by start time; fall back to file order if not in EmoEval
                entries.sort(key=lambda x: emo_info.get(x[0], {}).get("start", float("inf")))

                d_int = dialogue_counter
                id_map[d_int] = dialog_name
                dialogue_counter += 1

                for u_idx, (utt_str_id, text) in enumerate(entries):
                    info = emo_info.get(utt_str_id, {})
                    rows.append({
                        "Dialogue_ID":  d_int,
                        "Utterance_ID": u_idx,
                        "Utterance":    text,
                        "Speaker":      _speaker_from_utt_id(utt_str_id),
                        "Utt_Str_ID":   utt_str_id,
                        "Valence":      info.get("valence", np.nan),
                        "EmoCategory":  info.get("emotion", ""),
                    })

    df_all = pd.DataFrame(rows)

    # Split by session: train IDs are those whose dialog name starts with Ses0{1-4}
    train_mask = df_all["Utt_Str_ID"].str.startswith(
        tuple(f"Ses0{s}" for s in IEMOCAP_TRAIN_SESSIONS)
    )
    train_df = df_all[train_mask].reset_index(drop=True)
    test_df  = df_all[~train_mask].reset_index(drop=True)

    empty_dev_df = pd.DataFrame(columns=df_all.columns)

    print(f"IEMOCAP loaded: {len(train_df)} train utterances, "
          f"{len(test_df)} test utterances across {dialogue_counter} dialogues.")

    return train_df, empty_dev_df, test_df, id_map


def add_iemocap_sentiment(df: pd.DataFrame, neg_thresh: float = 2.0,
                          pos_thresh: float = 3.0) -> pd.DataFrame:
    """
    Adds a MELD-style 'Sentiment' column (positive/negative/neutral) derived
    from the per-utterance dimensional 'Valence' rating (1-5 scale) that
    load_iemocap_dataframes extracts from EmoEvaluation. This lets
    build_contextual_text append a [positive]/[negative]/[neutral] token to
    the target utterance for IEMOCAP, the same way it does for MELD's
    Sentiment column. Rows with missing valence get '' (no token appended).

    Thresholds are inclusive: valence <= neg_thresh -> 'negative',
    valence >= pos_thresh -> 'positive', otherwise 'neutral'.
    """
    df = df.copy()

    def bucket(v):
        if pd.isna(v):
            return ""
        if v <= neg_thresh:
            return "negative"
        if v >= pos_thresh:
            return "positive"
        return "neutral"

    df["Sentiment"] = df["Valence"].apply(bucket) if "Valence" in df.columns else ""
    return df


def iemocap_valence_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Diagnostic: mean/std/count of Valence per EmoCategory, to sanity-check
    that the neg/pos thresholds in add_iemocap_sentiment separate the
    emotion categories sensibly (e.g. hap/exc should skew high, ang/sad/fru
    low, neu in between).
    """
    if "Valence" not in df.columns or "EmoCategory" not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby("EmoCategory")["Valence"]
        .agg(["count", "mean", "std"])
        .sort_values("mean")
    )


def build_contextual_text(d_id: int, u_id: int, context_index: dict,
                          context_window: int, context_sep: str,
                          include_speaker: bool,
                          task_prefix: str = "",
                          use_sentiment_signal: bool = False) -> str:
    """
    Builds the encoder input for one utterance, prepending the previous
    `context_window` utterances from the same dialogue.

    Example (context_window=2, include_speaker=True, use_sentiment_signal=True):
      "Monica: What happened? [SEP] Ross: It was terrible [SEP] Rachel: I can't believe it [negative]"
    """
    utts = context_index.get(d_id, [])
    pos = next((i for i, (uid, _, _, _) in enumerate(utts) if uid == u_id), None)
    if pos is None:
        return task_prefix  # fallback: prefix only

    start = max(0, pos - context_window)
    window = utts[start: pos + 1]

    parts = []
    for i, (_, txt, spk, snt) in enumerate(window):
        is_target = (i == len(window) - 1)
        segment = f"{spk}: {txt}" if (include_speaker and spk) else txt
        if is_target and use_sentiment_signal and snt:
            segment = f"{segment} [{snt}]"
        parts.append(segment)

    body = context_sep.join(parts)
    return f"{task_prefix}{body}" if task_prefix else body