from collections.abc import Callable
from typing import Literal

from datasets import load_dataset
from transformers import AutoTokenizer

from tuner._typing import Chat, DatasetConfig
from tuner.data.dataset_sequence import ChatDataset
from tuner.datasets.dolly_ja_dataset import DollyJaMessageFormatter
from tuner.utils.distributed_utils import is_main_process


def get_dolly_ja_dataset_sequence(
    dataset_config: DatasetConfig,
    tokenizer: AutoTokenizer,
    *,
    format_fn: Callable[..., Chat] | None = None,
    model_max_length: int = 8192,
    split: Literal['train', 'eval', 'test'] = 'train',
    verbose: bool = is_main_process(),
) -> ChatDataset:
    """
    Returns tokenized dolly dataset sequence.

    Args:
        dataset_config (DatasetConfig)
        tokenizer (AutoTokenizer)
        format_fn (Callable[..., Message]): A function that takes strings and returns single formated prompt.
        model_max_length (int, defaults to `2048`): Max token length that model can handle including prompt and response.
        split (str, defaults to `train`): Which split to return.

    Returns:
        ChatDataset: preprocessed dataset of specified split

    Example:
        >>> from transformers import AutoTokenizer
        >>> import tuner.data import prepare_dataset

        >>> # instantiate tokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
        >>> tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        >>> tokenizer.eos_token = '<|eot_id|>'

        >>> # get the dataset
        >>> ds = prepare_dataset(
        >>>     'dolly_ja_dataset_sequence',
        >>>     tokenizer=tokenizer,
        >>>     model_max_length=8192,
        >>>     split='train,eval',
        >>> )
        >>> tokenizer.decode(ds['train'][0]['input_ids'])
    """

    if format_fn is None:
        format_fn = DollyJaMessageFormatter()

    dolly_ja = load_dataset('kunishou/databricks-dolly-15k-ja')['train']

    # split into train and test
    dolly_ja = dolly_ja.train_test_split(
        test_size=dataset_config.test_size, shuffle=True, seed=dataset_config.random_state)
    train_ds, test_ds = dolly_ja['train'], dolly_ja['test']

    # split inot train eval
    train_ds = train_ds.train_test_split(
        test_size=dataset_config.eval_size, shuffle=True, seed=dataset_config.random_state)
    train_ds, eval_ds = train_ds['train'], train_ds['test']

    ds = {'train': train_ds, 'eval': eval_ds, 'test': test_ds}[split]
    ds = ds.map(lambda x: {'chat': format_fn(**x)}, desc='Converting to chat format..')

    return ChatDataset(
        ds,
        tokenizer,
        model_max_length=model_max_length,
        split=split,
        user_header=dataset_config.user_header,
        assistant_header=dataset_config.assistant_header,
        shuffle=dataset_config.shuffle,
        inference_mode=dataset_config.inference_mode,
        use_subset=dataset_config.use_subset,
        subset_size=getattr(dataset_config, f'subset_{split}_size'),
        subset_seed=dataset_config.random_state,
        name=dataset_config.name,
    )
