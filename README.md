# PG-CLIP

Official implementation of **PG-CLIP: Periodicity-Guided Semantic--Visual
Adaptation for Cross-Process Few-Shot Defect Inspection in Semiconductor-
Display Manufacturing**.

[Paper](paper/PG-CLIP.pdf) · [Citation](CITATIONS.bib) ·
[Third-party notices](NOTICE)

PG-CLIP combines a source-trained text adapter, Adaptive Periodic Semantic
Fusion (APSF), and KAN-Residual visual adapters. The release also includes the
matched AA-CLIP controlled baseline and Generic-Prompt ablation.

## Installation

Python 3.10 and PyTorch 2.3.1 were used for the paper experiments.

```bash
conda env create -f environment.yml
conda activate pg-clip
```

Alternatively, install the pinned packages with `pip install -r
requirements.txt` in a clean environment.

## OpenAI CLIP checkpoint

Weights are not included. Download the official OpenAI CLIP ViT-L/14@336px
checkpoint and save it as `model/ViT-L-14-336px.pt`:

```bash
curl -L \
  https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt \
  -o model/ViT-L-14-336px.pt
sha256sum model/ViT-L-14-336px.pt
```

Expected SHA-256:
`3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02`
(filename `ViT-L-14-336px.pt`, 934,088,680 bytes).

## Data

The AMOLED Active/Cell dataset is proprietary and is not included. Directory
schemas, anonymous JSONL examples, and the official-MVTec conversion procedure
are documented in [docs/DATASETS.md](docs/DATASETS.md).

Prepare the public carpet/grid proxies directly from an official MVTec AD
download:

```bash
python scripts/prepare_mvtec_periodic.py \
  --mvtec-root /path/to/mvtec_anomaly_detection \
  --output-root data/MVTec-Periodic --mode symlink
```

## Split generation

AMOLED:

```bash
python scripts/build_amoled_directional_splits.py \
  --data-root data/AMOLED-ARRAY \
  --output-root splits/amoled_directional_fulltarget_v2 \
  --seeds 0,1,9 --shots 2,4,8,16,32
```

MVTec AD carpet/grid:

```bash
python scripts/build_mvtec_carpet_grid_splits.py \
  --data-root data/MVTec-Periodic \
  --output-root splits/mvtec_carpet_grid \
  --seeds 0,1,9 --shots 2,4,8,16,32
```

All subsets are nested within a fixed 16-normal/16-anomalous source pool.
Target metadata is used only after adaptation. The runner rejects any resolved
source/target image overlap.

## Reproduce one experiment

The following trains and evaluates AA-CLIP, Generic Prompts, APSF, and PG-CLIP:

```bash
python scripts/run_pgclip_protocol.py \
  --data-root data \
  --direction active_to_cell \
  --seed 0 --shot 2 --gpu 0
```

Directions are `active_to_cell`, `cell_to_active`, `carpet_to_grid`, and
`grid_to_carpet`. Add `--visualize` for anomaly maps or `--dry-run` to inspect
commands without accessing weights or data.

The two adaptation stages are explicit. Stage 1 trains the Generic-Prompt text
adapter and learns only APSF's scalar source-only fusion weight. Stage 2 reuses
the byte-identical text checkpoint and trains KAN-Residual visual adapters
after frozen visual Transformer blocks 1--6. Stage-1 segmentation supervision
uses the deepest selected CLIP feature (block 24), matching the experiments.

## Full benchmark and aggregation

Run both transfer directions, seeds 0/1/9, and 2/4/8/16/32 shots:

```bash
python scripts/run_full_benchmark.py \
  --data-root data --benchmark all --gpus 0,1
```

Aggregate completed runs:

```bash
python scripts/summarize_results.py \
  --results-root results/pgclip_protocol \
  --output-prefix results/summary
```

## Tests

```bash
python -m unittest discover -s tests -v
python train.py --help >/dev/null
python test.py --help >/dev/null
python scripts/run_full_benchmark.py --data-root data --dry-run
```

The tests cover APSF gradients, KAN-Residual initialisation, anomaly scoring,
text-checkpoint sharing, and target-data isolation. GitHub Actions runs the
same checks in a clean Python 3.10/PyTorch 2.3.1 CPU environment.

## License and attribution

PG-CLIP is released under Apache-2.0. It derives from
[AA-CLIP](https://github.com/Mwxinnn/AA-CLIP), contains code adapted from
[OpenAI CLIP](https://github.com/openai/CLIP) and
[OpenCLIP](https://github.com/mlfoundations/open_clip), and uses
[pykan](https://github.com/KindXiaoming/pykan). See [NOTICE](NOTICE) and
[CITATIONS.bib](CITATIONS.bib) for licenses and citations.

Copyright 2026 Chen Wei-Wei, Liao Yinping, and Leong Wai Yie.
