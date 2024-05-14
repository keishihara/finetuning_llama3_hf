import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias, TypeVar

import pandera as pa
from pandera.engines.pandas_engine import PydanticModel
from pydantic import BaseModel, field_validator
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from tuner.configs.dataset_configs import DatasetConfig

FilePath = str | os.PathLike[str]

TruncationSide = Literal['left', 'right', 'auto']

DatasetConfigDict = dict[str, DatasetConfig]

DatasetConfigs: TypeAlias = Sequence[DatasetConfig]
DatasetName: TypeAlias = str
DatasetNames: TypeAlias = Sequence[DatasetName]
DatasetConfigOrName: TypeAlias = DatasetConfig | DatasetName
DatasetConfigsOrNames: TypeAlias = DatasetConfig | DatasetConfigs | DatasetName | DatasetNames

T = TypeVar('T')
SplitLiteral: TypeAlias = Literal['train', 'eval', 'test']

valid_splits: set = {'train', 'eval', 'test', 'full'}

@dataclass
class Split:
    value: str
    _i: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not self._validate(self._value):
            msg = f'Invalid split value: {self._value}'
            raise ValueError(msg)

    def _validate(self, value: str) -> bool:
        value = re.sub(r'\s+', '', value)
        if ',' in value and '+' in value:
            msg = f'You cannot specify `,` and `+` in the split argument at the same time: {value}'
            raise ValueError(msg)

        splits = re.split(r',|\+', value)
        if not all(s in valid_splits for s in splits):
            msg = f'Invalid split value: {value}'
            raise ValueError(msg)

        return value

    @property
    def value(self) -> str:
        return self._value

    @value.setter
    def value(self, value: str) -> str:
        self._value = self._validate(value)

    def __len__(self) -> int:
        return len(re.split(r',|\+', self._value))

    def __iter__(self) -> 'Split':
        return self

    def __next__(self) -> str:
        splits = re.split(r',|\+', self._value)
        if self._i == len(splits):
            self._i = 0
            raise StopIteration
        split = splits[self._i]
        self._i += 1
        return split

GetDatasetFnCallable: TypeAlias = Callable[[
    DatasetConfig,
    AutoTokenizer,
    Callable,#  | BasePrompt,
    int,
    Split,
    bool,
], Dataset]

class Message(BaseModel):
    role: Literal['user', 'assistant', 'system']
    content: str

    @field_validator('role', mode='before')
    @classmethod
    def check_role(cls, x: T) -> T:
        if x not in {'system', 'user', 'assistant'}:
            raise ValueError(f'Invalid role: {x}')
        return x

Chat: TypeAlias = list[Message]

class ChatRecord(BaseModel):
    chat: Chat

class ChatSchema(pa.DataFrameModel):
    class Config:
        dtype = PydanticModel(ChatRecord)
        coerce = True
