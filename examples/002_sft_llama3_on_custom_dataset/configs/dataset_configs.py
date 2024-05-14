from dataclasses import dataclass
from pathlib import Path

from tuner.configs.dataset_configs import DatasetConfig

DATASET_PY_FOLDER = Path(__file__).parents[1] / 'custom_datasets'


@dataclass
class oasst2_ja_dataset_filtered(DatasetConfig):
    name: str = 'oasst2_ja_dataset_filtered'
    dataset_py_file: str = str(DATASET_PY_FOLDER / 'oasst2_ja_dataset_filtered.py:get_oasst2_ja_dataset_sequence')
    shuffle: bool = True
    unique_eval_frac: float = 0.05
    random_state: int = 42
