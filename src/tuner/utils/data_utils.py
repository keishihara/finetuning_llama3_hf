import json
from pathlib import Path

import pandas as pd

from tuner._typing import FilePath
from tuner.utils.distributed_utils import print_warning_on_rank_0


def is_json_serializable(obj: object) -> bool:
    try:
        json.dumps(obj)
    except TypeError:
        return False
    else:
        return True


def to_json_serializable(
    data_dict: dict,
    cls_to_exclude: tuple | None = None,
    keys_to_exclude: tuple | None = None,
) -> dict:

    if cls_to_exclude is None:
        cls_to_exclude = ()

    if keys_to_exclude is None:
        keys_to_exclude = ()

    new_data_dict = {}
    for k, v in data_dict.items():
        if not isinstance(v, cls_to_exclude) and k not in keys_to_exclude and isinstance(v, dict):
            try:
                new_data_dict[k] = json.dumps(v)
            except TypeError:
                print_warning_on_rank_0(f'Failed to save a dict object with key {k}: {v}')
        elif not isinstance(v, cls_to_exclude) and k not in keys_to_exclude and is_json_serializable(v):
            new_data_dict[k] = v
        elif hasattr(v, '__len__'):
            length = len(v)
            new_data_dict[k] = f'{type(v)} (len={length})'
        elif isinstance(v, Path):
            new_data_dict[k] = f'{v!s}'
        else:
            new_data_dict[k] = f'{type(v)}'

    return new_data_dict


def load_json(path: FilePath) -> dict:
    with Path.open(path) as f:
        return json.load(f)


def train_test_split_df(
    df: pd.DataFrame,
    *,
    test_size: float = 0.1,
    random_state: int = 42,
    drop_index: bool = True,
) -> tuple[pd.DataFrame, ...]:
    """
    Split a DataFrame into train and test subsets randomly.
    """
    if isinstance(test_size, int):
        df_test = df.sample(n=test_size, random_state=random_state, replace=False)
    elif isinstance(test_size, float):
        df_test = df.sample(frac=test_size, random_state=random_state, replace=False)
    else:
        msg = '`test_size` must be in either int or float.'
        raise TypeError(msg)
    indices = list(set(df.index.tolist()) - set(df_test.index.tolist()))
    df_train = df.loc[indices]
    return (
        df_train.reset_index(drop=drop_index),
        df_test.reset_index(drop=drop_index),
    )
