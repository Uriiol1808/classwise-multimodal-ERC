# `multimodal/`

Multimodal model built on top of the SDT architecture, extended with class-wise reliability-aware fusion, a valence-arousal circumplex prior, supervised contrastive alignment, and modality dropout. Consumes the per-utterance embeddings produced by `unimodal/` (see [`../unimodal/README.md`](../unimodal/README.md)).

## Files

| File | Description |
|---|---|
| `dataloader.py` | Loads precomputed text/audio/visual embeddings per conversation, builds batches with speaker and positional information, and applies modality dropout / degradation at load time when enabled. |
| `model.py` | SDT backbone (intra-/inter-modal transformers, hierarchical gated fusion, self-distillation) plus the additions: class-wise reliability-aware fusion, supervised contrastive head, and valence-arousal prior correction applied at inference. |
| `train.py` | Training entry point (task + self-distillation + contrastive losses). |
| `inference.py` | Runs a trained checkpoint on a dataset split (or single conversation) and applies the VA prior correction to the frozen model's logits. |

## Usage

```bash
# Train + Inference
python train.py \
  --dataset iemocap \
  --features_dir /path/to/unimodal/features/iemocap \
  --use_dropout \
  --checkpoint_dir checkpoints/iemocap

python train.py \
  --dataset meld \
  --features_dir /path/to/unimodal/features/meld \
  --use_contrastive \
  --checkpoint_dir checkpoints/meld
```

> Flag names above are illustrative — check `argparse` definitions in `train.py` / `inference.py` for the exact names before running.

## Configuration notes

- **Dataset-specific stacking**: IEMOCAP is trained with modality dropout + VA prior (no contrastive loss); MELD is trained with the contrastive loss + VA prior (no dropout). Uniform modality dropout is not well suited to MELD, where the fusion gate assigns dominant weight to text regardless of audio/visual condition.
- **VA prior is not learned**: `--va_alpha` and `--va_tau` are found by hyperparameter sweep over a frozen, already-trained model — the correction is applied only in `inference.py`, with no gradient updates.
- **Fusion strategies** in `model.py` cover softmax, entropy-based, modality-wise, and class-wise reliability weighting; class-wise fusion operates at the logit level, not on hidden representations.
- **Visual features**: expects `vit_action_units` (828-d) for MELD or `vit_landmarks` (1176-d) for IEMOCAP, matching the variant produced by `unimodal/visual/`.

## Expected input format

`dataloader.py` expects one embedding file per modality per conversation (or a packed archive — adjust to match your actual on-disk layout), aligned by utterance index with the dataset's original label files.