from collections.abc import Callable, Sequence
from dataclasses import is_dataclass
from pathlib import Path
from typing import NamedTuple

from torch.utils.data import Dataset
from transformers import AutoTokenizer

from tuner._typing import (
    DatasetConfig,
    DatasetConfigOrName,
    DatasetConfigsOrNames,
    GetDatasetFnCallable,
    Split,
)
from tuner.configs import dataset_configs as dataset_configs_module
from tuner.data import ConcatDataset
from tuner.utils import (
    load_module_from_py_file,
    print_on_rank_0,
)


class DatasetBuildingBlock(NamedTuple):
    dataset_name: str # predefined dataset name or user-defined custom dataset name
    dataset_config: DatasetConfig
    format_fn: Callable | None
    get_dataset_fn: GetDatasetFnCallable


def get_predefined_dataset_complete_set(
    dataset_config_or_name: DatasetConfigOrName,
) -> DatasetBuildingBlock:
    """
    Returns a DatasetBuildingBlock object of a predefined dataset available in llm/datasets.
    """
    if isinstance(dataset_config_or_name, str):
        # load dataset config from dataset_configs
        dataset_config = vars(dataset_configs_module).get(dataset_config_or_name)
        dataset_name = dataset_config_or_name
        if not dataset_config:
            msg = (
                f'Dataset config `{dataset_config_or_name}` is not found at '
                f'{vars(dataset_configs_module)["__file__"]}'
            )
            raise ValueError(msg)
        dataset_config = dataset_config()
    else:
        dataset_config = dataset_config_or_name
        dataset_name = dataset_config_or_name.name

    # load format_fn from formatter_py_file if available
    format_fn = None
    if hasattr(dataset_config, 'formatter_py_file') and dataset_config.formatter_py_file is not None:
        formatter_py_file = str(dataset_config.formatter_py_file)
        if ':' in formatter_py_file:
            module_path, cls_name = formatter_py_file.split(':')
        else:
            module_path, cls_name = formatter_py_file, 'PromptTemplate'
        module_path = Path(module_path)
        if module_path.suffix != '.py':
            msg = f'Provided file {module_path} is not a python (.py) file.'
            raise ValueError(msg)
        if not module_path.is_file():
            msg = f'Module {module_path.as_posix()} does not exist or is not a file.'
            raise FileNotFoundError(msg)
        formatter_module = load_module_from_py_file(module_path)
        format_fn = getattr(formatter_module, cls_name)()

    # load get_dataset_fn from DATASET_PRESETS
    from tuner.datasets import DATASET_PRESETS
    get_dataset_fn = DATASET_PRESETS.get(dataset_name)
    if get_dataset_fn is None:
        msg = f'Dataset name is not found: {dataset_name}'
        raise ValueError(msg)

    return DatasetBuildingBlock(dataset_name, dataset_config, format_fn, get_dataset_fn)


def get_custom_dataset_complete_set(dataset_config: DatasetConfig) -> DatasetBuildingBlock:
    """
    Returns a DatasetBuildingBlock object of a user-defined custom dataset.
    """

    dataset_py_file = str(dataset_config.dataset_py_file)
    if ':' in dataset_py_file:
        dataset_module_path, func_name = dataset_py_file.split(':')
    else:
        dataset_module_path, func_name = dataset_py_file, 'get_custom_dataset'
    if dataset_module_path.find('{version}') != -1:
        dataset_module_path = dataset_module_path.format(version=dataset_config.version)
    dataset_module_path = Path(dataset_module_path)
    if dataset_module_path.suffix != '.py':
        msg = f'Provided file {dataset_module_path} in {dataset_config} is not a python (.py) file.'
        raise ValueError(msg)
    if not dataset_module_path.is_file():
        msg = f'Dataset py file {dataset_module_path.as_posix()} does not exist or is not a file.'
        raise FileNotFoundError(msg)
    dataset_module = load_module_from_py_file(dataset_module_path)
    get_dataset_fn = getattr(dataset_module, func_name)

    if hasattr(dataset_module, 'PromptTemplate'):
        format_fn = dataset_module.PromptTemplate()

    elif hasattr(dataset_config, 'formatter_py_file') and dataset_config.formatter_py_file is not None:
        formatter_py_file = str(dataset_config.formatter_py_file)
        if ':' in formatter_py_file:
            formatter_module_path, cls_name = formatter_py_file.split(':')
        else:
            formatter_module_path, cls_name = formatter_py_file, 'PromptTemplate'
        if formatter_module_path.find('{version}') != -1:
            formatter_module_path = formatter_module_path.format(version=dataset_config.version)
        formatter_module_path = Path(formatter_module_path)
        if formatter_module_path.suffix != '.py':
            msg = f'Provided file {formatter_module_path} is not a python (.py) file.'
            raise ValueError(msg)
        if not formatter_module_path.is_file():
            msg = f'Module {formatter_module_path.as_posix()} does not exist or is not a file.'
            raise FileNotFoundError(msg)
        formatter_module = load_module_from_py_file(formatter_module_path)
        format_fn = getattr(formatter_module, cls_name)()
    else:
        format_fn = None

    return DatasetBuildingBlock(dataset_config.name, dataset_config, format_fn, get_dataset_fn)


def get_dataset_building_blocks(
    dataset_configs_or_names: DatasetConfigsOrNames,
) -> list[DatasetBuildingBlock]:
    """
    Returns a list of DatasetBuildingBlock objects given dataset_config(s) or dataset_name(s).
    """
    from tuner.datasets import DATASET_PRESETS

    dataset_building_blocks = []

    if not isinstance(dataset_configs_or_names, str) \
       and isinstance(dataset_configs_or_names, Sequence):
        for dataset_config_or_name in dataset_configs_or_names:
            if isinstance(dataset_config_or_name, str):
                res = get_predefined_dataset_complete_set(dataset_config_or_name)
                dataset_building_blocks.append(res)
            elif is_dataclass(dataset_config_or_name):
                if dataset_config_or_name.name in DATASET_PRESETS:
                    res = get_predefined_dataset_complete_set(dataset_config_or_name)
                else:
                    res = get_custom_dataset_complete_set(dataset_config_or_name)
                dataset_building_blocks.append(res)
            else:
                msg = f'Invalid dataset configs: {dataset_config_or_name}'
                raise ValueError(msg)

    elif isinstance(dataset_configs_or_names, str):
        res = get_predefined_dataset_complete_set(dataset_configs_or_names)
        dataset_building_blocks.append(res)

    elif is_dataclass(dataset_configs_or_names):
        if dataset_configs_or_names.name in DATASET_PRESETS:
            res = get_predefined_dataset_complete_set(dataset_configs_or_names)
        else:
            res = get_custom_dataset_complete_set(dataset_configs_or_names)
        dataset_building_blocks.append(res)
    else:
        msg = f'Invalid dataset configs: {dataset_configs_or_names}'
        raise ValueError(msg)

    return dataset_building_blocks


def prepare_dataset(
    dataset_configs_or_names: DatasetConfigsOrNames,
    tokenizer: AutoTokenizer,
    model_max_length: int = 4096,
    split: str | Split = 'train',
    *,
    verbose: bool = True,
    format_fn: Callable | None = None,
) -> Dataset | dict[str, Dataset]:
    """
    Args:
        dataset_configs_or_names (Sequence[dataclass] | datacalss | Sequence[str] | str): dataclass
            object(s) or dataset name(s). Dataset name must be one of the preset datasets at tuner/datasets/.
        tokenizer (AutoTokenizer)
        model_max_length (int, optional): model max length
        split (Split, optional): Which split of the data to load.
        verbose (bool, optional): If True, print loading message.
        format_fn (Callable, optional): A format function to override the one loaded from specified
            module by dataset_config.

    Returns:
        Dataset | dict[str, Dataset]: dataset object(s) which can be anything that
            is passed to HF Trainer. Usually, a sequence object implementing __getitem__ and __len__.

    Example:
        >>> from transformers import AutoTokenizer
        >>> from tuner.data import prepare_dataset
        >>> tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
        >>> tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        >>> tokenizer.eos_token = '<|eot_id|>'
        >>> ds_dict = prepare_dataset(
        ...     ['dolly_ja_dataset', 'dolly_ja_dataset'], # double dolly_ja_dataset
        ...     tokenizer=tokenizer,
        ...     split='train,eval',
        ... )
        >>> ds_dict
        {'train': Dataset({
            features: ['input_ids', 'labels', 'attention_mask'],
            num_rows: 13411
        }), 'eval': Dataset({
            features: ['input_ids', 'labels', 'attention_mask'],
            num_rows: 1490
        })}
        >>> # Signle split
        >>> ds_train = prepare_dataset('dolly_ja_dataset_sequence', tokenizer=tokenizer, split='train')
        >>> ds_train
        ChatDataset(name=dolly_ja_dataset_sequence:train, split=train, length=13423shuffle=False, augmentation=False, truncation_side=right)
        >>> # ConcatDataset
        >>> ds_concat = prepare_dataset('dolly_ja_dataset_sequence', tokenizer, split='train+eval')
        >>> ds_concat
        ConcatDataset(length=14915, n_datasets=2, datasets=[ChatDataset(name=dolly_ja_dataset_sequence:train, split=train, length=13423shuffle=False, augmentation=False, truncation_side=right),ChatDataset(name=dolly_ja_dataset_sequence:eval, split=eval, length=1492shuffle=False, augmentation=False, truncation_side=right)])
    """

    if isinstance(split, str):
        split = Split(split)

    splits = list(split)

    dataset_building_blocks = get_dataset_building_blocks(dataset_configs_or_names)

    datasets = {}
    for s in splits:
        for name, cfg, formatter, func in dataset_building_blocks:
            if verbose:
                msg = f'Loading {name}..'
                info = f'split: {s}'
                msg += f' ({info})'
                print_on_rank_0(msg)

            dataset = func(
                dataset_config=cfg,
                tokenizer=tokenizer,
                format_fn=formatter if formatter else format_fn,
                model_max_length=model_max_length,
                split=s,
                verbose=verbose,
            )
            datasets[s] = [*datasets.get(s, []), dataset]

    if len(splits) == 1:
        s = splits[0]
        if len(datasets[s]) == 1:
            return datasets[s][0]
        return ConcatDataset(datasets[s], split=s, verbose=verbose)

    for s in splits:
        if len(datasets[s]) == 1:
            datasets[s] = datasets[s][0]
        else:
            datasets[s] = ConcatDataset(
                datasets[s],
                split=s,
                verbose=verbose if '+' not in split.value else False,
            )
    if '+' in split.value:
        # Concat across all splits
        datasets = ConcatDataset(
            [datasets[s] for s in splits],
            split=split.value,
            verbose=verbose,
        )
    return datasets
