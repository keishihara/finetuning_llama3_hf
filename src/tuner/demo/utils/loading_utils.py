from pathlib import Path
from typing import Literal

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


def load_model_and_tokenizer(
    model_name_or_path: str,
    quantization: Literal['4bit', '8bit'] | None = None,
    *,
    merge_and_unload: bool = False,
    device_map: str = 'auto',
    silent: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:

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
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        model.eval()

        if getattr(tokenizer, 'name_or_path', None) in [
            'meta-llama/Meta-Llama-3-8B-Instruct',
            'meta-llama/Meta-Llama-3-70B-Instruct',
        ]:
            tokenizer.eos_token = '<|eot_id|>'

    return model, tokenizer
