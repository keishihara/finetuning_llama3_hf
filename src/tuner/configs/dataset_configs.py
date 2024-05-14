from dataclasses import dataclass, field

from tuner import PACKAGE_HOME


@dataclass(repr=False)
class DatasetConfig:
    name: str = None
    dataset_py_file: str = None
    formatter_py_file: str = None
    version: str = None
    data_file: str = None
    full_split: str = 'full'
    train_split: str = 'train'
    eval_split: str = 'eval'
    test_split: str = 'test'
    user_header: str = '<|start_header_id|>user<|end_header_id|>\n\n'
    assistant_header: str = '<|start_header_id|>assistant<|end_header_id|>\n\n'
    augmentation: bool = False
    shuffle: bool = False
    inference_mode: bool = False
    eval_size: float = 0.1
    test_size: float = 100
    use_subset: bool = False
    subset_train_size: int = 16
    subset_eval_size: int = 8
    subset_test_size: int = 8
    random_state: int = 42

    def __new__(cls, *args, **kwargs) -> 'DatasetConfig':
        if DatasetConfig in (cls,):
            msg = 'You cannot instantiate abstract class.'
            raise TypeError(msg)
        return super().__new__(cls)


@dataclass
class dolly_ja_dataset(DatasetConfig):
    name: str = 'dolly_ja_dataset'
    formatter_py_file: str = field(default=PACKAGE_HOME / 'datasets/dolly_ja_dataset.py:DollyJaMessageFormatter')

@dataclass
class dolly_ja_dataset_sequence(DatasetConfig):
    name: str = 'dolly_ja_dataset_sequence'
    formatter_py_file: str = field(default=PACKAGE_HOME / 'datasets/dolly_ja_dataset_sequence.py:DollyJaMessageFormatter')

@dataclass
class oasst2_dataset_sequence(DatasetConfig):
    name: str = 'oasst2_dataset_sequence'
    langs: list[str] = field(default_factory=lambda: ['en', 'ja'])

@dataclass
class oasst2_ja_dataset_sequence(DatasetConfig):
    name: str = 'oasst2_ja_dataset_sequence'
