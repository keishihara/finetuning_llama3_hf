import os
import warnings

from configs.training_args import ScriptArguments
from transformers import HfArgumentParser
from tuner.sft import train
from tuner.utils import load_dataset_configs, load_module_from_py_file

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
warnings.simplefilter('ignore')

if __name__ == '__main__':
    parser = HfArgumentParser([ScriptArguments])
    script_args = parser.parse_args_into_dataclasses()[0]

    module = load_module_from_py_file(script_args.dataset_config_module)
    # load dataset configs from module
    dataset_configs_or_names = load_dataset_configs(
        script_args.dataset_names,
        module,
        keep_missing_as_string=True,
    )

    if script_args.do_train:
        train(script_args, dataset_configs_or_names)
