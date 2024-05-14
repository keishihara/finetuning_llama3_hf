import time
from pathlib import Path

import torch
from peft import (
    PeftConfig,
    PeftModel,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from typer import Argument, Option, Typer

app = Typer()


@app.command()
def main(
    ckpt_path: str = Argument(..., help='Path to the checkpoint directory.'),
    quantization: str= Option(None, help="Quantization type: '4bit' or '8bit'"),
    save_path: str= Option(None, help='Path to a directory to save the merged model.'),
    model_name: str = Option(None, help='Model name to be used as the folder name.'),
    max_shard_size: str = Option('5GB', help='The maximum size for a checkpoint before being sharded.'),
) -> None:
    if quantization not in (None, '4bit', '8bit'):
        msg = '`quantization` must be either `4bit` or `8bit`.'
        raise ValueError(msg)

    if not quantization and torch.cuda.is_available():
        # It was required to be run on CPU if you want to merge the adapter
        # with full scale model without quantization.
        msg = (
            'CUDA devices are available although `quantization` is not specified. '
            'You must pass CUDA_VISIBLE_DEVICES= to hide GPUs from torch.'
        )
        raise RuntimeError(msg)

    ckpt_path = Path(ckpt_path)
    model_name = model_name or f'{ckpt_path.name}_{time.strftime("%y%m%dT%H%M%S%z")}'
    if not save_path:
        save_path = ckpt_path.parents[1] / 'release' / model_name
    else:
        save_path = Path(save_path).resolve()
        if not save_path.is_dir():
            msg = '`save_path` does not exist.'
            raise RuntimeError(msg)
        save_path = save_path / model_name
    save_path.mkdir(exist_ok=True, parents=True)

    if quantization == '4bit':
        torch_dtype = torch.bfloat16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4', # NF4 Quantization (for higher precision)
            bnb_4bit_use_double_quant=True, # Double Quantization (when you have problems with memory)
            bnb_4bit_compute_dtype=torch_dtype,
        )
    elif quantization == '8bit':
        torch_dtype = torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        torch_dtype = torch.bfloat16
        bnb_config = None

    print('Loading a base model and the adapter..')
    peft_config = PeftConfig.from_pretrained(ckpt_path)
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path, # both adapters share the same base model
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        device_map='auto' if quantization else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path,
        padding_side='left',
    )
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(
        model,
        ckpt_path,
        device_map='auto' if quantization else None,
    )
    print('Merging the adapter with the base model..')
    merged_model = model.merge_and_unload(safe_merge=True, progressbar=True)

    print('Saving the merged model..')
    merged_model.save_pretrained(
        save_path,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    tokenizer.save_pretrained(save_path, safe_serialization=True)
    with Path.open(save_path / 'original_ckpt_path.txt', 'w') as f:
        f.write(str(ckpt_path))

    print(f'Saved to {save_path}')
    print('Done.')


if __name__ == '__main__':
    app()
