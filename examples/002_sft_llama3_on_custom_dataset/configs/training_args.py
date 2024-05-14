from dataclasses import dataclass, field
from pathlib import Path

from tuner.configs import training_args


@dataclass
class ScriptArguments(training_args.ScriptArguments):
    dataset_names: list[str] | None = field(default_factory=lambda: [
        'oasst2_ja_dataset_filtered', # custom dataset
    ])
    dataset_config_module: str = str(Path(__file__).parent / 'dataset_configs.py')
