import importlib
import json
import socket
import time
import types
from collections.abc import Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import torch

from tuner._typing import DatasetConfig, FilePath
from tuner.configs import dataset_configs as dataset_configs_module
from tuner.configs.training_args import ScriptArguments
from tuner.utils.data_utils import to_json_serializable
from tuner.utils.distributed_utils import (
    is_main_process,
    print_warning_on_rank_0,
)


def load_module_from_py_file(py_file: FilePath) -> object:
    """
    Load a module from a Python file that is not in the Python path.

    Args:
        py_file (FilePath): The path to the Python file to be loaded.

    Returns:
        object: The loaded module.
    """
    module_name = Path(py_file).name
    loader = importlib.machinery.SourceFileLoader(module_name, str(py_file))
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def load_dataset_configs(
    dataset_config_names: str | Sequence[str],
    module: types.ModuleType = dataset_configs_module,
    *,
    keep_missing_as_string: bool = False,
) -> list[str | DatasetConfig]:
    """
    Load dataset configurations from a module.

    Args:
        dataset_config_names (str | Sequence[str]): The name or sequence of names
            of the dataset configurations to load.
        module (types.ModuleType, optional): The module from which to load the
            dataset configurations. Defaults to dataset_configs_module.
        keep_missing_as_string (bool, optional): Whether to keep missing
            configuration names as strings. Defaults to False.

    Returns:
        list[str | DatasetConfig]: A list of loaded dataset configurations or
            configuration names if missing.

    Raises:
        AttributeError: If a configuration name is not found and
            `keep_missing_as_string` is False.
    """
    if isinstance(dataset_config_names, str):
        cfg = getattr(module, dataset_config_names, None)
        if not keep_missing_as_string and cfg is None:
            msg = f'dataset config not found: {dataset_config_names}'
            raise AttributeError(msg)
        if keep_missing_as_string and cfg is None:
            return [dataset_config_names]
        return [cfg()]

    dataset_configs = []
    for name in dataset_config_names:
        cfg = getattr(module, name, None)
        if not keep_missing_as_string and cfg is None:
            msg = f'dataset config not found: {name}'
            raise AttributeError(msg)
        if keep_missing_as_string and cfg is None:
            dataset_configs = [*dataset_configs, name]
            continue
        dataset_configs = [*dataset_configs, cfg()]
    return dataset_configs


def save_configs(save_path: FilePath, **kwargs: dict[str, 'dataclass']) -> None:
    """
    Save configuration data classes to JSON files in the specified directory.

    Args:
        save_path (FilePath): The directory where the configuration files will be
            written.
        **kwargs (dict[str, 'dataclass']): A dictionary of configuration names
            and data classes to save.

    Returns:
        None

    Raises:
        RuntimeError: If `save_path` is not a directory.

    Note:
        This function only runs in the main process. If called from a non-main
            process, it will return immediately.
    """
    if not is_main_process():
        return

    save_path = Path(save_path)

    if not save_path.is_dir():
        msg = '`save_path` must be a folder.'
        raise RuntimeError(msg)

    for k, v in kwargs.items():
        if is_dataclass(v):
            with Path.open(save_path / f'{k}.json', 'w') as f:
                data_dict = to_json_serializable(asdict(v))
                json.dump(data_dict, f, indent=4, ensure_ascii=False)
        else:
            print_warning_on_rank_0(
                f'Not saving `{k}` parameter as it is not a dataclass.')


def sanity_check_args(args: ScriptArguments) -> None:
    """
    Perform sanity checks on the script arguments and adjust settings accordingly.

    Args:
        args (ScriptArguments): The script arguments to check and modify.

    Raises:
        RuntimeError: If both `load_in_4bit` and `load_in_8bit` are enabled at
            the same time.

    Note:
        This function adjusts the `save_strategy` and `save_steps` to match
        `evaluation_strategy` and `eval_steps` if `load_best_model_at_end` is
        enabled. It also forces the appropriate settings for 4-bit or 8-bit
        quantization and prints warnings if needed.
    """
    if args.load_in_4bit and args.load_in_8bit:
        msg = "You can't enable `load_in_4bit` and `load_in_8bit` at the same time."
        raise RuntimeError(msg)

    if args.load_best_model_at_end:
        # Can we know from which epoch is the checkpoint loaded at the end?
        args.save_strategy = args.evaluation_strategy
        args.save_steps = args.eval_steps

    if args.load_in_4bit:
        print_warning_on_rank_0('Forced to set bf16=True when bf16 is supported and 4-bit quantization is enabled.')
        args.fp16, args.bf16 = not torch.cuda.is_bf16_supported(), torch.cuda.is_bf16_supported()
    else:
        print_warning_on_rank_0('Forced to set fp16=True when 8-bit quantization is enabled.')
        args.fp16, args.bf16 = True, False

    if args.seed is None:
        print_warning_on_rank_0(
            'Seeding is disabled for this run. '
            'Pass --seed argument followed by a numerical value to enable seeding.',
        )


def setup_logging_dir(
    script_args: ScriptArguments,
    makedirs: bool = is_main_process(),
    subdirs: Sequence[str] | None = None,
    prefix: str = 'auto',
    *,
    create_ckpt_dir: bool = True,
    add_timestamp: bool = True,
    add_hostname: bool = True,
) -> None:
    """
    Set up the logging directory structure and assign the paths to the
    corresponding attributes in the `script_args` parameter. Also sets the
    generated `run_name` to `script_args.run_name`.

    The directory structure created will be as follows:

    ```
    logs/ # <--- Original `script_args.output_dir`, which will be overwritten by
        | # the new `ckpts` directory path
        |- {prefix}_{timestamp}/ # <--- `script_args.logging_dir`
            |- ckpts/   # <--- `script_args.output_dir`
            |- subdir1/ # <--- `script_args.subdir1_dir`
            |- subdir2/ # <--- `script_args.subdir2_dir`
    ```

    Args:
        script_args (ScriptArguments): The arguments containing directory paths
            and settings.
        makedirs (bool): Whether to create the directories. Default is based on
            `is_main_process()`.
        subdirs (Sequence[str] | None): List of subdirectories to create under
            the logging directory. Default is None.
        prefix (str): Prefix for the run name. Default is 'auto', which uses the
            model name or path.
        create_ckpt_dir (bool): Whether to create a `ckpts` directory. Default
            is True.
        add_timestamp (bool): Whether to add a timestamp to the run name.
            Default is True.
        add_hostname (bool): Whether to add the hostname to the run name.
            Default is True.

    Returns:
        None
    """

    timestamp, hostname = '', ''
    if add_timestamp:
        timestamp = '_' + time.strftime('%y%m%dT%H%M%S%z')
    if add_hostname:
        hostname = '_' + socket.gethostname()
    if not prefix:
        prefix = 'run'
    elif prefix == 'auto':
        prefix = script_args.model_name_or_path.replace('/', '--')
    run_name = f'{prefix}{timestamp}{hostname}'

    script_args.run_name = run_name
    script_args.logging_dir = Path(script_args.output_dir) /  run_name

    if makedirs:
        script_args.logging_dir.mkdir(exist_ok=True, parents=True)

    if create_ckpt_dir:
        script_args.output_dir = script_args.logging_dir / 'ckpts'
        if makedirs:
            script_args.output_dir.mkdir(exist_ok=True)

    if not isinstance(subdirs, Sequence):
        return

    for subdir in subdirs:
        subdir_ = subdir.replace('/','_') # make sure do not make nested folders
        setattr(script_args, subdir + '_dir', script_args.logging_dir / subdir_)
        if makedirs:
            subdir_ = getattr(script_args, f'{subdir_}_dir')
            subdir_.mkdir(exist_ok=True)
