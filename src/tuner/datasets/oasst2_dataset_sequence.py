from collections import defaultdict
from collections.abc import Callable
from typing import Any, Literal, TypeAlias

from datasets import (
    load_dataset,
)
from tqdm import tqdm
from transformers import AutoTokenizer

from tuner._typing import Chat, DatasetConfig
from tuner.data.dataset_sequence import ChatDataset
from tuner.utils import is_main_process

Thread: TypeAlias = list[dict[str, Any]]


def get_oasst2_dataset_sequence(
    dataset_config: DatasetConfig,
    tokenizer: AutoTokenizer,
    *,
    format_fn: Callable[..., Chat] | None = None,
    model_max_length: int = 8192,
    split: Literal['train', 'eval', 'test', 'full'] = 'train',
    verbose: bool = is_main_process(),
) -> ChatDataset:
    """
    Returns tokenized oasst2 ja dataset sequence.

    Args:
        dataset_config (DatasetConfig)
        tokenizer (AutoTokenizer)
        format_fn (Callable[..., Chat]): A function that takes strings and returns single formated prompt.
        model_max_length (int, defaults to `2048`): Max token length that model can handle including prompt and response.
        split (str, defaults to `train`): Which split to return.

    Returns:
        ChatDataset: preprocessed dataset of specified split

    Examples:
        >>> from transformers import AutoTokenizer
        >>> from tuner.data import prepare_dataset

        >>> # instantiate tokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
        >>> tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        >>> tokenizer.eos_token = '<|eot_id|>'

        >>> # get the dataset
        >>> ds = prepare_dataset(
        >>>     'oasst2_dataset_sequence',
        >>>     tokenizer=tokenizer,
        >>>     model_max_length=8192,
        >>>     split='train,eval', # or 'train+eval'
        >>> )
        >>> train_ds, _ = ds['train'], ds['eval']
        >>> train_ds
        ChatDataset(name=chat_dataset:train, length=13423, shuffle=False, augmentation=False, truncation_side=right)
        >>> tokenizer.decode(train_ds[0]['input_ids']) # input tensors
        >>> tokenizer.decode(train_ds[0]['labels'][train_ds[0]['labels'] != -100]) # labels
    """

    if split in ('train', 'eval'):
        ds = load_dataset('OpenAssistant/oasst2', split='train')
        # split into train and eval
        ds = ds.train_test_split(
            test_size=dataset_config.eval_size, shuffle=True, seed=dataset_config.random_state)
        ds = ds['test' if split == 'eval' else 'train']
    elif split == 'test':
        ds = load_dataset('OpenAssistant/oasst2', split='validation')
    else: # full
        ds = load_dataset('OpenAssistant/oasst2', split='train+validation')

    df = ds.to_pandas()
    df.role = df.role.replace({'prompter': 'user'})
    df = df.rename(columns={'text': 'content'})

    # prepare nodes and data_dict
    nodes = defaultdict(list)
    progbar = tqdm(
        df.to_dict(orient='records'),
        dynamic_ncols=True,
        desc='Building conversation tree..',
        disable=not is_main_process(),
    )
    for data in progbar:
        if data['parent_id']:
            nodes[data['parent_id']].append(data['message_id'])
    nodes = dict(nodes)

    data_dict = df.set_index('message_id').transpose().to_dict()

    # restore all message threads
    def follow(thread: Thread, current_id: str) -> list[Thread]:
        # Given a thread and a current_id, return the thread that follows the current_id
        thread = [*thread, {'message_id': current_id, **data_dict[current_id]}]
        if current_id in nodes:
            new_thread = []
            for child_id in nodes[current_id]:
                new_thread += follow(thread, child_id)
            return new_thread
        return [thread]

    def get_threads_from_root(root_id: str) -> list[Thread]:
        # Given a root_id, return all the threads in the tree
        all_threads = []
        thread = [{
            'message_id': root_id,
            **data_dict[root_id],
        }]
        for child_id in nodes.get(root_id, []):
            all_threads += follow(thread, child_id)
        return all_threads

    df_filtered = df.copy()
    if dataset_config.langs: # eg, ['ja', 'en']
        df_filtered = df[df.lang.isin(dataset_config.langs)]
    df_filtered = df_filtered[df_filtered.parent_id.isna()]

    tqdm.pandas(desc='Gathering threads..', dynamic_ncols=True, disable=not is_main_process())
    # make threads from root
    ser_thread = df_filtered.message_id.progress_apply(get_threads_from_root)
    ser_thread = ser_thread.explode().reset_index(drop=True)
    ser_thread = ser_thread[ser_thread.notna()]

    def cut_last_prompter_message(thread: Thread) -> Thread:
        if thread[-1]['role'] == 'user':
            return thread[:-1]
        return thread
    tqdm.pandas(desc='Cutting last user message..', dynamic_ncols=True, disable=not is_main_process())
    ser_thread = ser_thread.progress_apply(cut_last_prompter_message)

    df_chat = ser_thread.to_frame(name='thread')
    def to_chat(thread: list[dict]) -> Chat:
        return [{k: t[k] for k in ['role', 'content']} for t in thread]
    df_chat['chat'] = df_chat.thread.apply(to_chat)
    df_chat = df_chat[['chat']]

    return ChatDataset(
        df_chat,
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
