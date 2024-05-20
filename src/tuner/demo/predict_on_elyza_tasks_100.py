import warnings
from typing import Optional

import jsonlines
from datasets import load_dataset
from tqdm import tqdm
from transformers import TextStreamer
from typer import Typer
from utils.generation_utils import generate
from utils.loading_utils import load_model_and_tokenizer
from utils.logging_utils import setup_logging_dir_and_output_file

warnings.simplefilter('ignore')
app = Typer()

SYSTEM_PROMPT = 'You are a good AI assistant who always answers in natural Japanese.'
PROMPT_TEMPLATE = (
    '以下に、あるタスクを説明する指示があります。'
    'リクエストを適切に完了するための回答を記述してください。\n\n'
    '### 指示:\n{input}'
)
GENERATION_CONFIG = {
    'max_new_tokens': 1024,
    'use_cache': True,
    'do_sample': True,
    'top_p': 0.9,
    'temperature': 0.6,
    'repetition_penalty': 1.2,
}


def get_formatted_messages(data: dict) -> list[dict]:
    content = PROMPT_TEMPLATE.format(**data)
    messages = []
    if SYSTEM_PROMPT:
        messages.append({'role': 'system', 'content': SYSTEM_PROMPT})
    messages.append({'role': 'user', 'content': content})
    return messages


@app.command()
def predict(
    model_name_or_path: str,
    quantization: Optional[str] = None,   # noqa: UP007
    out_dir: str = 'auto',
    *,
    merge_and_unload: bool = False,
    silent: bool = False,
) -> None:

    out_path = setup_logging_dir_and_output_file(model_name_or_path, out_dir)
    if not silent:
        print(f'{out_path = !s}')

    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path,
        quantization=quantization,
        merge_and_unload=merge_and_unload,
        silent=silent,
    )
    streamer = TextStreamer(tokenizer, skip_prompt=True) if not silent else None
    dataset = load_dataset('elyza/ELYZA-tasks-100', split='test')

    with jsonlines.open(out_path, mode='w') as writer:
        for idx in tqdm(
            range(len(dataset)),
            desc='Predicting on test..',
            dynamic_ncols=True,
            disable=silent,
        ):
            messages = get_formatted_messages(dataset[idx])
            if not silent:
                print('--------------------')
                print(messages[1]['content'])
                print('>>>>')
            assistant_message = generate(
                messages,
                model=model,
                tokenizer=tokenizer,
                streamer=streamer,
                generation_config=GENERATION_CONFIG,
            )
            messages.append({'role': 'assistant', 'content': assistant_message})
            writer.write(messages)


if __name__ == '__main__':
    app()
