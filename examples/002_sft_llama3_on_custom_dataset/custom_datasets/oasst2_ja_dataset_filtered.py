import uuid
from collections import defaultdict
from collections.abc import Callable
from typing import Literal, TypeVar

import neologdn
import numpy as np
import pandas as pd
import pandera as pa
from datasets import load_dataset
from pandarallel import pandarallel
from pandera.dtypes import Int32
from pandera.typing import DataFrame
from tqdm import tqdm
from transformers import AutoTokenizer
from tuner._typing import ChatSchema, DatasetConfig, Message, SplitLiteral
from tuner.data.dataset_sequence import ChatDataset
from tuner.utils import is_main_process

try:
    from custom_datasets.utils import replace_eos
except ImportError:
    from utils import replace_eos

T = TypeVar('T')

pandarallel.initialize(progress_bar=False, nb_workers=8, verbose=0)
tqdm.pandas()


class Oasst2Schema(pa.DataFrameModel):
    message_id: str = pa.Field(unique=True)
    parent_id: str = pa.Field(nullable=True)
    user_id: str
    created_date: str = pa.Field(unique=True)
    text: str
    role: str = pa.Field(isin=['system', 'user', 'assistant'])
    lang: str = pa.Field(isin=['en', 'ja'])
    review_count: Int32 = pa.Field(gt=1)
    message_tree_id: str
    labels: dict
    content: str # text translated to Japanese

    class Config:
        strict = False

class ThreadSchema(pa.DataFrameModel):
    thread_id: str = pa.Field(unique=True)
    message_tree_id: str
    thread: list[dict]


@pa.check_types
def to_threads(df: DataFrame[Oasst2Schema]) -> DataFrame[ThreadSchema]:
    nodes = defaultdict(list)
    progbar = tqdm(df.to_dict(orient='records'),
                   dynamic_ncols=True,
                   desc='Building conversation tree..',
                   disable=True)
    for data in progbar:
        if data['parent_id']:
            nodes[data['parent_id']].append(data['message_id'])
    nodes = dict(nodes)

    def follow(thread: list[dict], current_id: str) -> list[list[dict]]:
        # Given a thread and a current_id, return the thread that follows the current_id
        thread = [*thread, {'message_id': current_id, **data_dict[current_id]}]
        if current_id in nodes:
            new_thread = []
            for child_id in nodes[current_id]:
                new_thread += follow(thread, child_id)
            return new_thread
        return [thread]

    def get_threads_from_root(root_id: str) -> list[list[dict]]:
        # Given a root_id, return all the threads in the tree
        all_threads = []
        thread = [{'message_id': root_id, **data_dict[root_id]}]
        for child_id in nodes[root_id]:
            all_threads += follow(thread, child_id)
        return all_threads

    data_dict = df.set_index('message_id').transpose().to_dict()

    df_filtered = df.copy()
    df_filtered = df_filtered[df_filtered.lang.isin(['en', 'ja'])]
    df_filtered = df_filtered[df_filtered.parent_id.isna()]

    ser_thread = df_filtered.message_id.apply(get_threads_from_root)
    ser_thread = ser_thread.explode().reset_index(drop=True)
    ser_thread = ser_thread[ser_thread.notna()]

    def cut_last_prompter_message(thread: list[dict]) -> list[dict]:
        if thread[-1]['role'] == 'prompter':
            return thread[:-1]
        return thread
    ser_thread = ser_thread.apply(cut_last_prompter_message)

    df_thread = ser_thread.to_frame(name='thread')
    df_thread['thread_id'] = df_thread.thread.apply(lambda _: str(uuid.uuid4()))
    df_thread['message_tree_id'] = df_thread.thread.str[0].str['message_tree_id']

    # sort by message_id
    df_thread['_full_message_id'] = df_thread.thread.apply(lambda xs: [x['message_id'] for x in xs]).str.join('_')
    df_thread = df_thread.sort_values(by='_full_message_id').reset_index(drop=True)
    df_thread = df_thread.drop(columns='_full_message_id')

    return df_thread[['thread_id', 'message_tree_id', 'thread']]


@pa.check_types
def load_oasst2_dataframe() -> DataFrame[Oasst2Schema]:
    # load oasst2 (japanese translation)
    oasst2_ja = load_dataset('kunishou/oasst2-135k-ja', split='train')
    df_ja = oasst2_ja.to_pandas()
    df_ja = df_ja.drop_duplicates(subset='message_id')

    # load original oasst2
    oasst2 = load_dataset('OpenAssistant/oasst2', split='train+validation')
    df = oasst2.to_pandas()

    df = df.merge(df_ja[['message_id', 'text_ja']])
    df.role = df.role.replace({'prompter': 'user'})
    df = df.rename(columns={'text_ja': 'content'})

    # English and Japanese only, reviewed multiple times, and drop entries without labels
    return df.query(r'lang in ["en", "ja"] & review_count > 1 & labels.notna()')


@pa.check_types
def get_message_ids_of_quality_data(df: DataFrame[Oasst2Schema]) -> list[str]:
    df_filtered = df.copy()
    tqdm.pandas(desc='converting labels..', dynamic_ncols=True, disable=not is_main_process())
    df_filtered.labels = df_filtered.labels.progress_apply(
        lambda x: pd.DataFrame(x).set_index('name').transpose().to_dict())

    filtering_conditions = [
        ('lang_mismatch', '==0.0'),
        ('spam', '==0.0'),
        ('fails_task', '==0.0'),
        ('pii', '==0.0'),
        ('not_appropriate', '==0.0'),
        ('hate_speech', '==0.0'),
        # ('sexual_content', ''),
        ('quality', '>=0.7'),
        ('toxicity', '<0.1'),
        # ('humor', ''),
        ('helpfulness', '>=0.7'),
        # ('creativity', ''),
        ('violence', '==0.0'),
    ]
    conds = [f'labels.str["{k}"].str["value"]{v}' for k, v in filtering_conditions]
    df_filtered = df_filtered.query(' & '.join(conds))

    return df_filtered.message_id.tolist()


def filter_thread(df_thread: pd.DataFrame) -> pd.DataFrame:
    def are_all_assistants_accepted(row: pd.Series) -> bool:
        is_assistant = [msg['role'] == 'assistant' for msg in row.thread]
        indices = np.where(is_assistant)[0].tolist()
        return all(np.array([msg['is_accepted'] for msg in row.thread])[indices])
    return df_thread[
        df_thread.apply(are_all_assistants_accepted, axis=1) # All assistant messages are accepted
        | (df_thread.thread.str.len().gt(4)) # or, thread length more than 4 (we want long context data)
    ].copy().reset_index(drop=True)


def normalize_assistant_messages(thread: list[dict]) -> list[dict]:
    for i in range(len(thread)):
        if thread[i]['role'] != 'assistant':
            continue
        content = thread[i]['content']
        content = replace_eos(content)
        # FIXME: This may break inline code blocks, links, short novels, markdown texts, and so on
        content = neologdn.normalize(content, tilde='ignore')
        thread[i]['content'] = content
    return thread


def get_preprocessed_oasst2_ja(
    unique_eval_frac: float = 0.05,
    split_seed: int = 42,
    split: Literal['train', 'eval'] | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    if split not in ('train', 'eval', None):
        raise ValueError(f'Currently `train` or `eval` is valid, got: {split}')

    df = load_oasst2_dataframe()
    message_ids = get_message_ids_of_quality_data(df)
    df['is_accepted'] = df.message_id.isin(message_ids)
    df_thread = to_threads(df)
    df_filtered = filter_thread(df_thread)
    df_filtered.thread = df_thread.thread.apply(normalize_assistant_messages)

    # train test split based on message_tree_id
    eval_message_tree_ids = (
        df
        .message_tree_id
        .drop_duplicates()
        .sample(frac=unique_eval_frac, random_state=split_seed)
    )
    train_df = df_filtered.query('~message_tree_id.isin(@eval_message_tree_ids)')
    eval_df = df_filtered.query('message_tree_id.isin(@eval_message_tree_ids)')
    df_dict = {
        'train': train_df.reset_index(drop=True),
        'eval': eval_df.reset_index(drop=True),
    }
    if split:
        return df_dict[split]
    return df_dict


def to_chat(thread: list[dict]) -> list[Message]:
    return [{k: t[k] for k in ['role', 'content']} for t in thread]


def get_oasst2_ja_dataset_sequence(
    dataset_config: DatasetConfig,
    tokenizer: AutoTokenizer,
    *,
    format_fn: Callable | None = None,
    model_max_length: int = 8192,
    split: SplitLiteral = 'train',
    verbose: bool = is_main_process(),
) -> ChatDataset:
    df = get_preprocessed_oasst2_ja(
        unique_eval_frac=dataset_config.unique_eval_frac,
        split_seed=dataset_config.random_state,
        verbose=verbose,
        split=split,
    )
    df['chat'] = df.thread.apply(to_chat)
    df = ChatSchema.validate(df)
    return ChatDataset(
        df,
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
