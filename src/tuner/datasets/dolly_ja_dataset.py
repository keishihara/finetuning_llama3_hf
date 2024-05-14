import warnings
from collections.abc import Callable
from typing import Literal

import numpy as np
from datasets import (
    Dataset,
    load_dataset,
)
from transformers import AutoTokenizer

from tuner._constants import IGNORE_INDEX
from tuner._typing import Chat, DatasetConfig
from tuner.prompts.base_prompt import FormatterMixin
from tuner.utils.distributed_utils import is_main_process, print_on_rank_0

NOMAXLENGTHWARNING = True


class DollyJaMessageFormatter(FormatterMixin):
    TEMPLATE_WITHOUT_CONTEXT = """
    以下の# Instructionはタスクを説明する指示です。要求を適切に満たすように回答してください。

    # Instruction
    {instruction}
    """

    TEMPLATE_WITH_CONTEXT = """
    以下の# Instructionはタスクを説明する指示で、# Contextはタスクに必要な情報です。
    要求を適切に満たすように回答してください。

    # Instruction
    {instruction}

    # Context
    {input}
    """

    def format(self, **x) -> Chat:
        if x.get('input'):
            user = self.dedent(self.TEMPLATE_WITH_CONTEXT).format(**x)
        else:
            user = self.dedent(self.TEMPLATE_WITHOUT_CONTEXT).format(**x)
        user = user.strip()
        assistant = x['output'].strip()
        return [
            {'role': 'user', 'content': user},
            {'role': 'assistant', 'content': assistant},
        ]


def preprocess(
    example: dict,
    tokenizer: AutoTokenizer,
    user_token_ids: list,
    assistant_token_ids: list,
    ignore_index: int = IGNORE_INDEX,
) -> dict:
    input_ids = example['formatted_chat_tokens'].detach().clone()
    labels = example['formatted_chat_tokens'].detach().clone()
    assistant_ignored = False
    assistant_token_ids_indices = []
    for assistant_idx in np.where(labels == assistant_token_ids[0])[0]:
        # find the indices of the start of a response.
        if (
            labels[assistant_idx : assistant_idx + len(assistant_token_ids)].tolist()
            == assistant_token_ids
        ):
            # tokens after the beginning header tokens are considered as assistant tokens
            assistant_token_ids_indices = [
                *assistant_token_ids_indices,
                assistant_idx + len(assistant_token_ids),
            ]

    if len(assistant_token_ids_indices) == 0:
        if not NOMAXLENGTHWARNING:
            warnings.warn(
                f'Could not find assistant key `{assistant_token_ids}` in the '
                f'following instance: {tokenizer.decode(input_ids)} '
                f'This instance will be ignored in loss calculation. '
                f'Note, if this happens often, consider increasing the `max_length`.',
                stacklevel=2,
            )
        labels[:] = ignore_index
        assistant_ignored = True

    user_token_ids_indices = []
    for user_idx in np.where(labels == user_token_ids[0])[0]:
        # find the indices of the start of a human answer.
        if (
            labels[user_idx : user_idx + len(user_token_ids)].tolist()
            == user_token_ids
        ):
            # user message including the beginning header tokens
            user_token_ids_indices = [*user_token_ids_indices, user_idx]

    if not assistant_ignored and len(user_token_ids_indices) == 0:
        if not NOMAXLENGTHWARNING:
            warnings.warn(
                f'Could not find user key `{user_token_ids}` in the '
                f'following instance: {tokenizer.decode(input_ids)} '
                f'This instance will be ignored in loss calculation. '
                f'Note, if this happens often, consider increasing the `max_length`.',
                stacklevel=2,
            )
        labels[:] = ignore_index

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
            labels[start:end] = ignore_index
        else:
            labels[:end] = ignore_index

    if len(assistant_token_ids_indices) < len(user_token_ids_indices):
        labels[user_token_ids_indices[-1] :] = ignore_index

    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': input_ids.ne(tokenizer.pad_token_id),
    }


def get_dolly_ja_dataset(
    dataset_config: DatasetConfig,
    tokenizer: AutoTokenizer,
    format_fn: Callable[..., Chat],
    *,
    model_max_length: int = 2048,
    split: Literal['train', 'eval', 'test'] = 'train',
    verbose: bool = is_main_process(),
) -> Dataset:
    """
    Returns tokenized dolly dataset(s).

    Args:
        dataset_config (DatasetConfig)
        tokenizer (AutoTokenizer)
        format_fn (Callable[..., Chat]): A function that takes strings and returns
            single formated prompt.
        model_max_length (int, defaults to `2048`): Max token length that model can
            handle including prompt and response.
        split (str, defaults to `train`): Which split to return.

    Returns:
        Dataset: preprocessed dataset of specified split

    Example:
        >>> from transformers import AutoTokenizer
        >>> import tuner.data import prepare_dataset

        >>> # instantiate tokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
        >>> tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        >>> tokenizer.eos_token = '<|eot_id|>'

        >>> # get the dataset
        >>> ds = prepare_dataset(
        >>>     'dolly_ja_dataset',
        >>>     tokenizer=tokenizer,
        >>>     model_max_length=8192,
        >>>     split='train,eval',
        >>> )
        >>> tokenizer.decode(ds['train'][0]['input_ids'])
    """

    dolly_ja = load_dataset('kunishou/databricks-dolly-15k-ja')['train']

    # split into train and test
    dolly_ja = dolly_ja.train_test_split(
        test_size=dataset_config.test_size, shuffle=True, seed=dataset_config.random_state)
    train_ds, test_ds = dolly_ja['train'], dolly_ja['test']

    # split inot train eval
    train_ds = train_ds.train_test_split(
        test_size=dataset_config.eval_size, shuffle=True, seed=dataset_config.random_state)
    train_ds, eval_ds = train_ds['train'], train_ds['test']

    ds = {
        'train': train_ds,
        'eval': eval_ds,
        'test': test_ds,
    }[split]

    ds = ds.map(
        lambda x: {'messages': format_fn(**x)},
        desc='Converting to messages format..',
    )
    ds = ds.map(
        lambda x: {'formatted_chat_tokens': tokenizer.apply_chat_template(
            x['messages'],
            add_generation_prompt=False,
            return_tensors='pt',
            truncation=True,
            max_length=model_max_length,
        ).squeeze()},
        desc='Applying chat template..',
    )
    ds.set_format(type='torch', columns=['formatted_chat_tokens'])

    user_token_ids = tokenizer.encode(dataset_config.user_header, add_special_tokens=False)
    assistant_token_ids = tokenizer.encode(dataset_config.assistant_header, add_special_tokens=False)
    ds = ds.map(
        preprocess,
        fn_kwargs={
            'tokenizer': tokenizer,
            'user_token_ids': user_token_ids,
            'assistant_token_ids': assistant_token_ids,
            'ignore_index': IGNORE_INDEX,
        },
        remove_columns=list(ds.features),
        desc='Forming input tokens..',
    )
    ds = ds.filter(
        lambda x: not (x['labels'] == IGNORE_INDEX).all(),
        desc='filtering non trainable examples..',
    )
    if dataset_config.use_subset:
        ds = ds.select(range(getattr(dataset_config, f'subset_{split}_size')))
    if verbose:
        print_on_rank_0(f'Dataset size: {split}={len(ds)}')
    return ds
