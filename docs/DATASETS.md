# Dataset preparation

## Proprietary AMOLED Active/Cell data

The AMOLED benchmark is proprietary and is not distributed with this
repository. PG-CLIP expects the following directory structure:

```text
<data-root>/
  AMOLED-ARRAY/
    Active/
      images/normal/
      images/anomaly/
      mask/anomaly/
    Cell/
      images/normal/
      images/anomaly/
      mask/anomaly/
```

Each anomalous image must have a same-stem PNG mask. Metadata uses relative
paths and the four fields shown in `examples/metadata/*.example.jsonl`.
Class labels may be anonymised identifiers; all unknown identifiers inherit
the process-level prompt name, so no internal product code is required.

The split builder intentionally does not encode private sample counts or
factory identifiers. Dataset owners must separately approve publication of
exact counts, internal class codes, filenames, and collection details. The
software license does not grant access to or redistribution rights for the
AMOLED data.

## Public MVTec AD carpet/grid proxy

Download [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
from its official website and retain the original category layout. Create the
two proxy domains without redistributing the dataset:

```bash
python scripts/prepare_mvtec_periodic.py \
  --mvtec-root /path/to/mvtec_anomaly_detection \
  --output-root data/MVTec-Periodic \
  --mode symlink
```

The converter combines each category's `train/good` and `test/good` images as
normal samples and uses all anomalous test images with their official masks.
It verifies the official carpet/grid counts. Use `--mode copy` when symbolic
links are unavailable.

Then generate the nested source subsets and complete target sets:

```bash
python scripts/build_mvtec_carpet_grid_splits.py \
  --data-root data/MVTec-Periodic \
  --output-root splits/mvtec_carpet_grid \
  --seeds 0,1,9 --shots 2,4,8,16,32
```

MVTec AD remains subject to its own terms and citation requirements.
