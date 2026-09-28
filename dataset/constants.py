"""Dataset and prompt constants for the paper protocols.

Data locations are derived from ``PGCLIP_DATA_ROOT`` or the repository-local
``data`` directory. No filesystem scanning occurs at import time.
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PGCLIP_DATA_ROOT", PROJECT_ROOT / "data"))

DATA_PATH = {
    "AMOLED-Active": str(DATA_ROOT / "AMOLED-ARRAY" / "Active"),
    "AMOLED-Cell": str(DATA_ROOT / "AMOLED-ARRAY" / "Cell"),
    "MVTec-Periodic-carpet": str(DATA_ROOT / "MVTec-Periodic" / "carpet"),
    "MVTec-Periodic-grid": str(DATA_ROOT / "MVTec-Periodic" / "grid"),
}

# Generic defaults. Evaluation classes are read from an explicitly supplied
# JSONL file, making them independent of local folders.
CLASS_NAMES = {
    "AMOLED-Active": ["panel"],
    "AMOLED-Cell": ["panel"],
    "MVTec-Periodic-carpet": ["carpet"],
    "MVTec-Periodic-grid": ["grid"],
}

DOMAINS = {dataset: "Industrial" for dataset in DATA_PATH}

REAL_NAMES = {
    "AMOLED-Active": {
        "panel": "amoled active array panel",
    },
    "AMOLED-Cell": {
        "panel": "amoled cell array panel",
    },
    "MVTec-Periodic-carpet": {"carpet": "carpet surface"},
    "MVTec-Periodic-grid": {"grid": "grid surface"},
}

PROMPTS = {
    "prompt_normal": ["{}", "a {}", "the {}"],
    "prompt_abnormal": [
        "a damaged {}", "a broken {}", "a {} with flaw",
        "a {} with defect", "a {} with damage",
    ],
    "prompt_templates": ["{}.", "a photo of {}."],
}

MVFA_PROMPTS = {
    "prompt_normal": [
        "{}", "flawless {}", "perfect {}", "unblemished {}",
        "{} without flaw", "{} without defect", "{} without damage",
    ],
    "prompt_abnormal": [
        "damaged {}", "broken {}", "{} with flaw", "{} with defect",
        "{} with damage",
    ],
    "prompt_templates": [
        "a bad photo of a {}.", "a low resolution photo of the {}.",
        "a bad photo of the {}.", "a cropped photo of the {}.",
        "a bright photo of a {}.", "a dark photo of the {}.",
        "a photo of my {}.", "a photo of the cool {}.",
        "a close-up photo of a {}.", "a black and white photo of the {}.",
        "a bright photo of the {}.", "a cropped photo of a {}.",
        "a jpeg corrupted photo of a {}.", "a blurry photo of the {}.",
        "a photo of the {}.", "a good photo of the {}.",
        "a photo of one {}.", "a close-up photo of the {}.",
        "a photo of a {}.", "a low resolution photo of a {}.",
        "a photo of a large {}.", "a blurry photo of a {}.",
        "a jpeg corrupted photo of the {}.", "a good photo of a {}.",
        "a photo of the small {}.", "a photo of the large {}.",
        "a black and white photo of a {}.", "a dark photo of a {}.",
        "a photo of a cool {}.", "a photo of a small {}.",
        "there is a {} in the scene.", "there is the {} in the scene.",
        "this is a {} in the scene.", "this is the {} in the scene.",
        "this is one {} in the scene.",
    ],
}

PERIODICITY_AWARE_PROMPTS = {
    "prompt_normal": [
        "a clean periodic pattern with continuous local structure",
        "a uniform periodic texture with consistent spacing",
        "a regular repeated pattern without local interruption",
        "a periodic visual structure with stable local repetition",
        "a clean repetitive texture with preserved local regularity",
        "a repeated pattern with no contaminant mark",
    ],
    "prompt_abnormal": [
        "a periodic pattern with a local structural disruption",
        "a repeated pattern with a local periodic interruption",
        "a repeated pattern with a locally missing element",
        "a repeated pattern with a locally shifted element",
        "a repeated pattern with uneven local spacing",
        "a repeated pattern with broken local continuity",
        "a repeated pattern with a distorted local unit",
        "a regular local structure with broken periodic continuity",
        "a periodic pattern with a local contamination mark",
        "a periodic texture with a small stain disrupting local regularity",
        "a repeated pattern with a residue-like spot",
        "a periodic visual structure with a foreign spot",
        "a clean repeated texture interrupted by local contamination",
        "a repeated texture with a local stain-like region",
        "a periodic texture with residue breaking local continuity",
        "a repeated pattern with a small dirty spot",
        "a periodic structure partially covered by contamination",
        "a repetitive texture with an isolated intensity anomaly",
        "a periodic pattern with local contrast inconsistency",
        "a repeated visual structure with a small irregular blob",
        "a periodic texture crossed by a thin contaminant trace",
        "a regular periodic background with a localized surface stain",
    ],
    "prompt_templates": [
        "{}.", "a close-up image of {}.", "a cropped image of {}.",
        "a high resolution inspection image of {}.",
    ],
}
