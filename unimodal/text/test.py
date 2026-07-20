"""
Run on bl-dacomsrv:
  python iemocap_diag.py /path/to/IEMOCAP_full_release
Checks for files/lines silently dropped by main_text.py's parsers.
"""
import sys
from pathlib import Path
from models import (
    _TRANS_LINE_RE, _parse_transcription, _parse_emo_eval,
    load_iemocap_dataframes, IEMOCAP_TRAIN_SESSIONS, IEMOCAP_TEST_SESSIONS,
)

root = Path(sys.argv[1])

# 1) lines that don't match the transcription regex (excluding blanks/comments)
for sess in IEMOCAP_TRAIN_SESSIONS + IEMOCAP_TEST_SESSIONS:
    trans_dir = root / f"Session{sess}" / "dialog" / "transcriptions"
    for f in sorted(trans_dir.glob("*.txt")):
        if f.name.startswith("._"):
            continue
        with open(f, "r", errors="replace") as fh:
            for ln, line in enumerate(fh, 1):
                s = line.strip()
                if not s or s.startswith("//"):
                    continue
                if not _TRANS_LINE_RE.match(s):
                    print(f"UNMATCHED LINE {f.name}:{ln}: {s!r}")

        entries = _parse_transcription(f)
        if not entries:
            print(f"DROPPED DIALOGUE (0 entries): {f.name}")

# 2) dialogues with missing EmoEvaluation (order falls back to file order)
for sess in IEMOCAP_TRAIN_SESSIONS + IEMOCAP_TEST_SESSIONS:
    trans_dir = root / f"Session{sess}" / "dialog" / "transcriptions"
    emo_dir   = root / f"Session{sess}" / "dialog" / "EmoEvaluation"
    for f in sorted(trans_dir.glob("*.txt")):
        if f.name.startswith("._"):
            continue
        if not (emo_dir / f.name).exists():
            print(f"NO EMOEVAL FILE: {f.name}")

# 3) totals
train_df, dev_df, test_df, id_map = load_iemocap_dataframes(str(root))
print(f"\nDialogues: {len(id_map)} (expected 151)")
print(f"Train utterances: {len(train_df)} (expected 5810)")
print(f"Test utterances:  {len(test_df)} (expected 1623)")