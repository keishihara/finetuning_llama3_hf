import json
import time
import warnings
from pathlib import Path
from typing import Any, Literal

import torch
from datasets import load_dataset
from peft import (
    PeftConfig,
    PeftModel,
)
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer,
)
from typer import Typer

warnings.simplefilter('ignore')
app = Typer()

# enable_full_determinism(42, warn_only=True) # disabled for optimized cuda computation

SYSTEM_PROMPT = 'You are an honest and talented AI assistant who always answers in natural Japanese.'
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
}


def setup_logging_dir_and_output_file(model_name_or_path: str, out_dir: str = 'auto') -> str:
    """
    Set up logging directory and output file for predictions.

    Args:
        model_name_or_path (str): The name or path of the model.
        out_dir (str): The output directory for saving predictions.

    Returns:
        str: Path to the output file for predictions.
    """
    timestamp = time.strftime('%y%m%dT%H%M%S%z')

    if out_dir == 'auto':
        if Path(model_name_or_path).is_dir(): # pretrained model checkpoint
            out_dir = Path(model_name_or_path) / 'predictions'
            local_or_hf = 'local_model'
        else: # hf model id
            out_dir = Path(__file__).parent / 'predictions' / model_name_or_path.replace('/', '--')
            local_or_hf = model_name_or_path.replace('/', '--')
    else:
        out_dir = Path(out_dir)

    out_file = f'predictions_{local_or_hf}_{timestamp}.json'
    out_dir.mkdir(exist_ok=True, parents=True)
    return out_dir / out_file


class PredictionWriter:
    """
    A class to handle writing prediction results to a file.

    Args:
        file_path (str): The path to the output file.
        meta_data (Any): Metadata to be included in the file.
        overwirte (bool): Whether to overwrite the file if it exists.
    """
    def __init__(self, file_path: str, meta_data: Any = None, *, overwirte: bool = False) -> None:
        self.file_path = file_path
        self.file = None
        self.content = None
        self.dump_kwargs = {'indent': 4, 'ensure_ascii': False}

        if not overwirte and Path(file_path).is_file():
            raise FileExistsError(f'Provided `file_path` already exists: {file_path}')

        with Path.open(file_path, 'w') as f:
            # Initialize empty dicts for storing meta_data and messages
            if meta_data:
                json.dump({'meta_data': meta_data, 'messages': {}}, f, **self.dump_kwargs)
            else:
                json.dump({'messages': {}}, f, **self.dump_kwargs)

    def __enter__(self) -> 'PredictionWriter':
        self.file = Path.open(self.file_path, 'r+')
        self.file.seek(0)
        self.content = json.load(self.file)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._save_to_file()
        if self.file:
            self.file.close()

    def log(self, index: int | str, messages: list[dict]) -> None:
        """
        Log a new prediction result.

        Args:
            index (int | str): The index of the prediction.
            messages (list[dict]): The messages to be logged.
        """
        if str(index) in self.content['messages']:
            raise ValueError(f'Messages already exist at `{index = }`.')
        self.content['messages'][str(index)] = messages
        self._save_to_file()

    def _save_to_file(self) -> None:
        """Save the current content to the file."""
        if self.file:
            self.file.seek(0)
            json.dump(self.content, self.file, **self.dump_kwargs)
            self.file.truncate()


def load_model_and_tokenizer(
    model_name_or_path: str,
    quantization: Literal['4bit', '8bit'] | None = None,
    device_map: str = 'auto',
    *,
    merge_and_unload: bool = False,
    silent: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the model and tokenizer.

    Args:
        model_name_or_path (str): The name or path of the model.
        quantization (Literal['4bit', '8bit'] | None): The quantization type.
        device_map (str): The device map for model loading.
        merge_and_unload (bool): Whether to merge and unload the model.
        silent (bool): Whether to suppress print statements.

    Returns:
        tuple[AutoModelForCausalLM, AutoTokenizer]: The loaded model and tokenizer.
    """
    if quantization == '4bit':
        if not silent:
            print('4bit quantization (NF4) specified.')
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
    elif quantization == '8bit':
        if not silent:
            print('8bit quantization (LLM.int8()) specified.')
        torch_dtype = torch.float16
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        if not silent:
            print('No quantization specified.')
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        bnb_config = None

    is_peft_model = (Path(model_name_or_path) / 'adapter_config.json').is_file()
    if is_peft_model:
        if not silent:
            print(f'Pretrained LoRA Adapter found: {model_name_or_path}')
        peft_config = PeftConfig.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path, # both adapters share the same base model
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation='flash_attention_2',
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side='left',
        )
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(
            model,
            model_name_or_path,
            device_map='auto',
        )
        if merge_and_unload:
            model = model.merge_and_unload(progressbar=True, safe_merge=True)
        model.eval()

    else:
        if not silent:
            print(f'Loading {model_name_or_path}..')
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation='flash_attention_2',
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side='left',
        )
        tokenizer.pad_token = '[PAD]'
        model.resize_token_embeddings(len(tokenizer))
        model.eval()

        if getattr(tokenizer, 'name_or_path', None) in [
            'meta-llama/Meta-Llama-3-8B-Instruct',
            'meta-llama/Meta-Llama-3-70B-Instruct',
        ]:
            tokenizer.eos_token = '<|eot_id|>'

    return model, tokenizer


def get_formatted_messages(data: dict) -> list[dict]:
    """
    Args:
        data (dict): The input data for the message, which must have a value
            with `input` key.

    Returns:
        list[dict]: A list of formatted messages.
    """
    content = PROMPT_TEMPLATE.format(**data)
    messages = []
    if SYSTEM_PROMPT:
        messages.append({'role': 'system', 'content': SYSTEM_PROMPT})
    messages.append({'role': 'user', 'content': content})
    return messages


@torch.inference_mode()
def generate(
    messages: list[dict],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    streamer: TextStreamer | None = None,
    generation_config: dict | None = None,
) -> str:
    """
    Generate a response based on the input messages.

    Args:
        messages (list[dict]): The input messages for generation.
        model (AutoModelForCausalLM): The loaded model for generation.
        tokenizer (AutoTokenizer): The loaded tokenizer for generation.
        streamer (TextStreamer | None): The text streamer for real-time output.
        generation_config (dict | None): The configuration for generation.

    Returns:
        str: The decoded response text.
    """
    if not generation_config:
        generation_config = {}
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors='pt',
    )
    input_ids = input_ids.to(model.device)
    outputs = model.generate(
        input_ids,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        streamer=streamer,
        **generation_config,
    )
    outputs = outputs[0][len(input_ids[0]):]
    return tokenizer.decode(outputs, skip_special_tokens=True)


@app.command()
def predict(
    model_name_or_path: str,
    quantization: str | None = None,
    out_dir: str = 'auto',
    *,
    merge_and_unload: bool = False,
    silent: bool = False,
) -> None:
    """
    Predict and generate responses for the ELYZA-tasks-100 dataset.

    Args:
        model_name_or_path (str): The name or path of the model.
        quantization (Optional[str]): The quantization type.
        out_dir (str): The output directory for saving predictions.
        merge_and_unload (bool): Whether to merge and unload the model.
        silent (bool): Whether to suppress print statements.
    """

    out_path = setup_logging_dir_and_output_file(model_name_or_path, out_dir)
    if not silent:
        print(f'{out_path = !s}')

    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path,
        quantization=quantization,
        merge_and_unload=merge_and_unload,
    )
    streamer = TextStreamer(tokenizer, skip_prompt=True) if not silent else None
    dataset = load_dataset('elyza/ELYZA-tasks-100', split='test')

    meta_data = {
        'model_name_or_path': model_name_or_path,
        'quantization': quantization,
        'merge_and_unload': merge_and_unload,
        'system_prompt': SYSTEM_PROMPT,
        'prompt_template': PROMPT_TEMPLATE,
        'generation_config': GENERATION_CONFIG,
    }

    with PredictionWriter(out_path, meta_data=meta_data, overwirte=True) as writer:
        for idx in tqdm(
            range(len(dataset)),
            desc='Predicting on test..',
            dynamic_ncols=True,
            disable=silent,
        ):
            messages = get_formatted_messages(dataset[idx])
            if not silent:
                print('--------------------')
                print(next(m['content'] for m in messages if m['role'] == 'user'))
                print('>>>>')
            assistant_message = generate(
                messages,
                model=model,
                tokenizer=tokenizer,
                streamer=streamer,
                generation_config=GENERATION_CONFIG,
            )
            messages.append({'role': 'assistant', 'content': assistant_message})
            writer.log(idx, messages)


if __name__ == '__main__':
    app()
