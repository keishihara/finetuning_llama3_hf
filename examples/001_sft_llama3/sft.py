import os
import warnings
from dataclasses import dataclass, field

from transformers import HfArgumentParser
from tuner.configs import training_args
from tuner.sft import train

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
warnings.simplefilter('ignore')

@dataclass
class ScriptArguments(training_args.ScriptArguments):
    dataset_names: list[str] | None = field(default_factory=lambda: [
        'dolly_ja_dataset', # one of preset datasets
        'oasst2_dataset_sequence', # one of preset datasets
    ])


if __name__ == '__main__':
    parser = HfArgumentParser([ScriptArguments])
    script_args = parser.parse_args_into_dataclasses()[0]

    if script_args.do_train:
        train(script_args, script_args.dataset_names)
