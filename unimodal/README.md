# `unimodal/`

Feature extraction for each modality. Run these before `multimodal/` — outputs are the per-utterance embeddings consumed by `multimodal/dataloader.py`.

## `text/`

| File | Description |
|---|---|
| `main.py` | Extracts text embeddings with RoBERTa-large (1024-d) for each utterance. |
| `models.py` | Text encoder wrapper. |

```bash
python text/main.py --dataset iemocap --data_dir /path/to/iemocap --out_dir features/iemocap/text
```

## `audio/`

| File | Description |
|---|---|
| `main.py` | Extracts audio embeddings with emotion2vec_plus_large (1024-d) per utterance. |
| `models.py` | Audio encoder wrapper. |
| `degrade_audio.py` | Applies controlled acoustic degradation (noise / corruption) for robustness training and evaluation. |

```bash
python audio/main.py --dataset iemocap --data_dir /path/to/iemocap --out_dir features/iemocap/audio

# optional: generate degraded audio variants for robustness experiments
python audio/degrade_audio.py --data_dir /path/to/iemocap --out_dir features/iemocap/audio_degraded
```

## `visual/`

| File | Description |
|---|---|
| `asd.py` | Active speaker detection (Light-ASD), used to isolate the speaking face per utterance. |
| `main.py` | Extracts appearance + geometry features: ViT (768-d) appearance embeddings combined with 3D facial landmarks (3DDFA-V2, ONNX). Produces the `vit_landmarks` variant (1176-d), used for IEMOCAP. |
| `models.py` | Encoder(s) used by `main.py`. |
| `main_au.py` | Extracts action-unit features via py-feat (60-d) combined with ViT appearance. Produces the `vit_action_units` variant (828-d), used for MELD. Requires the separate `au_env` (py-feat + scipy compatibility patches). |
| `models_au.py` | Encoder(s) used by `main_au.py`. |
| `utils.py` | Face detection (MTCNN) and shared preprocessing utilities. |
| `degrade_video.py` | Applies controlled visual degradation (blur / occlusion / dropout of frames) for robustness training and evaluation. |

```bash
# landmark-based variant (IEMOCAP)
conda activate sdt_env
python visual/main.py --dataset iemocap --data_dir /path/to/iemocap --out_dir features/iemocap/visual

# action-unit variant (MELD) — separate env for py-feat
conda activate au_env
python visual/main_au.py --dataset meld --data_dir /path/to/meld --out_dir features/meld/visual

# optional: degraded visual variants for robustness experiments
python visual/degrade_video.py --data_dir /path/to/iemocap --out_dir features/iemocap/visual_degraded
```

> Flag names above are illustrative — check `argparse` definitions in each `main*.py` before running.

## Notes

- Run `visual/asd.py` before `visual/main.py` / `visual/main_au.py` if the pipeline requires pre-cropped speaking-face tracks as input; otherwise it's called internally — check the script.
- Feature extraction is per-dataset: use IEMOCAP's landmark-based visual variant and MELD's action-unit-based visual variant to match the configuration expected by `multimodal/`.
- Output embeddings must be aligned by utterance index with the dataset's label files for `multimodal/dataloader.py` to load correctly.