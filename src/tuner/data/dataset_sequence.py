from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch
from datasets import Dataset as HfDataset
from torch.utils.data import ConcatDataset as ConcatDatasetTorch
from torch.utils.data import Dataset

from tuner._constants import IGNORE_INDEX
from tuner._typing import ChatSchema
from tuner.utils.distributed_utils import (
    is_main_process,
    print_on_rank_0,
    print_warning_on_rank_0,
)

if TYPE_CHECKING:
    from typing import Literal

    from pandera.typing import DataFrame
    from transformers import AutoTokenizer


class ChatDataset(Dataset):
    def __init__(
        self,
        dataset: DataFrame | HfDataset,
        tokenizer: AutoTokenizer,
        *,
        model_max_length: int = 8192,
        split: Literal['train', 'eval', 'test'] = 'train',
        user_header: str | list[int] = '<|start_header_id|>user<|end_header_id|>\n\n',
        assistant_header: str | list[int] = '<|start_header_id|>assistant<|end_header_id|>\n\n',
        format_fn: Callable | None = None,
        shuffle: bool = False,
        inference_mode: bool = False,
        ignore_index: bool = IGNORE_INDEX,
        truncation_side: Literal['left', 'right'] = 'right',
        verbose: bool = is_main_process(),
        use_subset: bool = False,
        subset_size: bool = 16,
        subset_seed: int = 42,
        name: str | None = None,
    ) -> None:
        """
        Args:
            dataset (DataFrame | datasets.Dataset): source dataset which is expected to have
                a field named `chat` where chat messages are stored. Otherwise, you need to format
                an item from this dataset using custom format_fn callable object.
            tokenizer (AutoTokenizer)
            model_max_length (int, optional): maximum length of the input sequence.
                Defaults to 4096.
            split (Literal['train', 'eval', 'test', 'full'], optional): split of
                the dataset. Defaults to 'train'.
            user_header (str | list[int]): A text which user message starts with.
                Defaults to the Llama3 user message header.
            assistant_header (str | list[int]): A text which assistant message starts with
                Defaults to the Llama3 assistant message header.
            format_fn (Callable, optional): prompt formatter object that takes two positional args,
                and returns formatted message chain.
            shuffle (bool, optional): whether to shuffle the dataset. Defaults to False.
            inference_mode (bool, optional): whether to set the dataset in inference
                mode. if True, input_ids and attention_mask are set to the prompt only,
                not the whole example. Defaults to False.
            ignore_index (int, optional): index to be ignored in loss calculation.
                Defaults to -100.
            truncation_side (Literal['left', 'right'], optional): from which side the input tensor
                will be truncated. Defaults to `right`.
            verbose (bool, optional): whether to print verbose messages.
                Defaults to True.
            use_subset (bool, optional): whether to use a subset of the dataset.
                Defaults to False.
            subset_size (int, optional): size of the subset. Defaults to 16.
            subset_seed (int, optional): the random state used when randomly sampling the subset.
                Defaults to 42.
            name (str, optional): name of this ChatDataset instance.
                Defaults to 'chat_dataset:{split}'.

        Examples:
            >>> from datasets import load_dataset
            >>> from transformers import AutoTokenizer
            >>> from tuner.datasets.dolly_ja_dataset import DollyJaMessageFormatter

            >>> tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
            >>> tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            >>> tokenizer.eos_token = '<|eot_id|>'

            >>> dolly_ds = load_dataset('kunishou/databricks-dolly-15k-ja', split='train')
            >>> format_fn = DollyJaMessageFormatter() # format each row to [{'role': 'user', 'content': 'user prompt'}, ...]
            >>> dolly_ds = dolly_ds.map(lambda x: {'chat': format_fn(**x)}, desc='Converting to chat format..')
            >>> chat_dataset = ChatDataset(dataset=dolly_ds, tokenizer=tokenizer)
            >>> chat_dataset
            ChatDataset(name=chat_dataset:train, split=train, length=15015, shuffle=False, augmentation=False, truncation_side=right)
            >>> chat_dataset[0].keys()
            dict_keys(['input_ids', 'labels', 'attention_mask'])
            >>> chat_dataset.count_tokens()
            {'total': 4415221, 'user+system': 2695091, 'assistant': 1720130}
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.split = split
        self.user_header = user_header
        self.assistant_header = assistant_header
        self.shuffle = shuffle
        self.inference_mode = inference_mode
        self.ignore_index = ignore_index
        self.truncation_side = truncation_side
        self.verbose = verbose
        self.use_subset = use_subset
        self.subset_size = subset_size
        self.name = f'{name}:{split}' if name else f'chat_dataset:{split}'

        dataset: DataFrame = dataset.to_pandas() if isinstance(dataset, HfDataset) else dataset
        self.dataset = ChatSchema.validate(dataset)

        if isinstance(format_fn, Callable):
            self.format_fn = format_fn
        else:
            self.format_fn = lambda x, *_: x['chat']

        if self.split == 'test' and not self.inference_mode:
            print_warning_on_rank_0('Forced to set `inference_mode=True` for test split.')
            self.inference_mode = True

        if self.split in ('train', 'eval') and self.inference_mode:
            print_warning_on_rank_0(
                f'You are setting `inference_mode=True` for {self.split} split.\n'
                f'If this is not intended, please set `inference_mode=False` (eg, finetuning).',
            )
            self.inference_mode = True

        if hasattr(self.tokenizer, 'truncation_side'):
            self.tokenizer.truncate_side = truncation_side

        if isinstance(user_header, str):
            self.user_token_ids = self.tokenizer.encode(
                self.user_header, add_special_tokens=False)
        else:
            self.user_token_ids = self.user_header

        if isinstance(assistant_header, str):
            self.assistant_token_ids = self.tokenizer.encode(
                self.assistant_header, add_special_tokens=False)
        else:
            self.assistant_token_ids = self.assistant_header

        self.max_length_warning_done_once = True # set to False to actually turn on this feature
        self.index_list = None
        self.__augmentation = False

        # select subset if specified
        if self.use_subset:
            if len(self.dataset) < self.subset_size:
                msg = (
                    f'Length of the dataset ({len(self.dataset)}) is smaller than the '
                    f'specified subset size ({self.subset_size}).'
                )
                raise ValueError(msg)
            self.dataset = self.dataset.sample(self.subset_size, random_state=subset_seed)

        # initialize index_list
        if self.shuffle:
            self.index_list = torch.randperm(len(self.dataset))
        else:
            self.index_list = torch.arange(len(self.dataset))

        if self.verbose:
            print_on_rank_0(
                f'ChatDataset {self.name} initialized (length={len(self.dataset):,}, '
                f'shuffle={self.shuffle})',
            )

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(name={self.name}, split={self.split}, length={len(self)}, '
            f'shuffle={self.shuffle}, augmentation={self.augmentation}, truncation_side={self.truncation_side})'
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: str) -> dict:
        index = self.index_list[index].item()
        return self.get_single_datapoint(index)

    @property
    def augmentation(self) -> bool:
        return self.__augmentation

    @augmentation.setter
    def augmentation(self, aug: bool) -> None:
        """Enable or disable augmentaion property by setting
        dataset.augmentation = True/False.
        """
        if aug:
            self.init_aug_dataset()

        if self.verbose:
            if aug:
                print_on_rank_0(f'[{self.name}] Augmentation enabled.')
                if self.split in ('eval', 'test'):
                    print_on_rank_0(
                        f'You are setting augmentation=True to {self.split} set. '
                        f'It is not recommended, and make sure you set random state.',
                    )
            else:
                print_on_rank_0(f'[{self.name}] Augmentation disabled.')

        self.__augmentation = aug

    def count_tokens(self, *, disable_pbar: bool = False) -> dict[str, int]:
        from tqdm import tqdm
        progbar = tqdm(range(len(self)),
                       dynamic_ncols=True,
                       desc='Counting samples..',
                       disable=not is_main_process() or disable_pbar)
        total, trainable = 0, 0
        for idx in progbar:
            sample = self[idx]
            total += len(sample['input_ids'])
            trainable += len(sample['labels'][sample['labels'] != IGNORE_INDEX])
        return {
            'total': total,
            'user+system': total - trainable,
            'assistant': trainable,
        }

    def get_random_idx(self) -> int:
        if not self.__is_initialized:
            msg = 'dataset has not been initialized yet.'
            raise RuntimeError(msg)

        return torch.randint(0, len(self), (1,)).item()

    def get_single_datapoint(self, index: int) -> dict:
        if self.inference_mode:
            raise NotImplementedError

        chat_messages = self.get_single_chat_thread(index)
        input_ids = self.tokenizer.apply_chat_template(
            chat_messages,
            add_generation_prompt=False,
            return_tensors='pt',
            truncation=True,
            max_length=self.model_max_length,
        )
        input_ids = input_ids.detach().clone().squeeze()
        labels = input_ids.detach().clone()
        labels = self._mark_user_tokens_not_trainable(labels)
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
        }

    def _mark_user_tokens_not_trainable(self, labels: torch.Tensor) -> torch.Tensor:
        """This method ensures that all tokens of the labels are set to an `ignore_index`
        when they do not come from the assistant. This way, the loss is only calculated
        on the completion made by the assistant.
        """
        if labels.ndim > 1:
            msg = f'Batched operation not yet supported. (got a tensor of shape {labels.shape})'
            raise NotImplementedError(msg)

        assistant_ignored = False
        assistant_token_ids_indices = []
        for assistant_idx in np.where(labels == self.assistant_token_ids[0])[0]:
            # find the indices of the start of a response.
            if (
                labels[assistant_idx : assistant_idx + len(self.assistant_token_ids)].tolist()
                == self.assistant_token_ids
            ):
                # tokens after the beginning header tokens are considered as assistant tokens
                assistant_token_ids_indices = [
                    *assistant_token_ids_indices,
                    assistant_idx + len(self.assistant_token_ids),
                ]

        if len(assistant_token_ids_indices) == 0:
            labels[:] = self.ignore_index
            assistant_ignored = True

        user_token_ids_indices = []
        for user_idx in np.where(labels == self.user_token_ids[0])[0]:
            # find the indices of the start of a human answer.
            if (
                labels[user_idx : user_idx + len(self.user_token_ids)].tolist()
                == self.user_token_ids
            ):
                # user message including the beginning header tokens
                user_token_ids_indices = [*user_token_ids_indices, user_idx]

        if not assistant_ignored and len(user_token_ids_indices) == 0:
            labels[:] = self.ignore_index

        if (
            len(user_token_ids_indices) > 0
            and len(assistant_token_ids_indices) > 0
            and user_token_ids_indices[0] > assistant_token_ids_indices[0]
        ):
            user_token_ids_indices = [0, *user_token_ids_indices]

        for idx, (start, end) in (
            enumerate(zip(user_token_ids_indices, assistant_token_ids_indices, strict=False))
        ):
            # Make pytorch loss function ignore all non response tokens
            if idx != 0:
                labels[start:end] = self.ignore_index
            else:
                labels[:end] = self.ignore_index

        if len(assistant_token_ids_indices) < len(user_token_ids_indices):
            labels[user_token_ids_indices[-1] :] = self.ignore_index

        return labels

    def get_single_chat_thread(self, index: int) -> list[dict]:
        """Returns text messages of single chat thread.
        chat = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
            {"role": "user", "content": "I'd like to show off how chat templating works!"},
        ]
        """
        single_datapoint = self.dataset.iloc[index]
        return self.format_fn(single_datapoint, self)


class ConcatDataset(ConcatDatasetTorch):
    def __init__(
        self,
        datasets: list[Dataset | HfDataset],
        *,
        split: str | None = None,
        shuffle: bool = False,
        verbose: bool = True,
    ) -> None:
        self.datasets = datasets
        self.split = split
        self.verbose = verbose
        self.length = sum([len(ds) for ds in self.datasets])
        self.index_to_dataset_mapping = []
        for dataset in self.datasets:
            for j in range(len(dataset)):
                self.index_to_dataset_mapping.append((dataset, j))

        # initialize index_list
        if shuffle:
            self.index_list = torch.randperm(self.length)
        else:
            self.index_list = torch.arange(self.length)

        if self.verbose:
            print_on_rank_0(f'ConcateDataset initialized (split={self.split}, length={self.length:,}, shuffle={shuffle})')

    def __len__(self) -> int:
        return self.length

    def __repr__(self) -> str:
        datasets = ','.join([repr(ds) for ds in self.datasets])
        return (
            f'{self.__class__.__name__}(length={self.length}, '
            f'n_datasets={len(self.datasets)}, datasets=[{datasets}])'
        )

    def __getitem__(self, index: int) -> dict:
        if not isinstance(index, int):
            msg = f'`index` must be integer, got: {index}'
            raise TypeError(msg)

        if index >= len(self):
            msg = '`index` out of range.'
            raise IndexError(msg)

        index_ = self.index_list[index]
        dataset, j = self.index_to_dataset_mapping[index_]
        return dataset[j]

    def get_random_idx(self) -> int:
        return torch.randint(0, len(self), (1,)).item()

    @property
    def augmentation(self) -> list[tuple[str, bool]]:
        return [(
            ds.name if hasattr(ds, 'name') else i,
            ds.augmentation if hasattr(ds, 'augmentation') else 'augmentation property does not exist.',
        ) for i, ds in enumerate(self.datasets, 1)]

    @augmentation.setter
    def augmentation(self, aug: bool) -> None:
        """Enable or disable augmentaion property by setting
        dataset.augmentation = True/False.
        """
        for ds in self.datasets:
            if hasattr(ds, 'augmentation'):
                ds.augmentation = aug
            elif self.verbose:
                print_warning_on_rank_0(
                    f'Augmentation property does not exist in the dataset: {ds}.',
                )
