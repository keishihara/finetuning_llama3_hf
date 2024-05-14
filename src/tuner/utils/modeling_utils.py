import re

import numpy as np
import torch
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch import nn
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.utils.quantization_config import QuantizationConfigMixin

from tuner.configs.training_args import ScriptArguments
from tuner.utils.distributed_utils import is_main_process, print_on_rank_0, wait_for_everyone
from tuner.utils.tokenizer_utils import smart_tokenizer_and_embedding_resize


def get_quantization_config_and_dtype(args: ScriptArguments) -> tuple[QuantizationConfigMixin, torch.dtype]:
    """
    Determine the quantization configuration and data type based on the script arguments.

    Args:
        args (ScriptArguments): The script arguments containing quantization settings.

    Returns:
        tuple[QuantizationConfigMixin, torch.dtype]: A tuple containing the quantization
            configuration and the torch data type.
    """
    if args.load_in_4bit:
        print_on_rank_0('4-bit QLoRA enabled.')
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
    else:
        print_on_rank_0('8-bit QLoRA enabled.')
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        torch_dtype = torch.float16

    return bnb_config, torch_dtype


def find_target_modules(model: nn.Module, excludes: str | list[str] | None = 'lm_head') -> list[str]:
    """
    Find target modules in the model, excluding specified ones.

    Args:
        model (nn.Module): The model to search for target modules.
        excludes (str | list[str] | None): Module names or patterns to exclude. Default is 'lm_head'.

    Returns:
        list[str]: A list of target module names sorted alphabetically.
    """
    if isinstance(excludes, list):
        excludes = '|'.join(excludes)
    target_modules = list({
        name.split('.')[-1]
        for name, module in model.named_modules()
        if 'Linear' in str(type(module))
    })
    if excludes:
        target_modules = [module for module in target_modules if not re.search(excludes, module)]
    return sorted(target_modules)


def load_model_and_tokenizer(
    args: ScriptArguments,
    quantization_config: QuantizationConfigMixin,
    torch_dtype: torch.dtype,
    *,
    print_summary: bool = True,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the model and tokenizer based on the provided arguments and configuration.

    Args:
        args (ScriptArguments): The script arguments containing model and tokenizer settings.
        quantization_config (QuantizationConfigMixin): The quantization configuration to use.
        torch_dtype (torch.dtype): The torch data type to use.
        print_summary (bool, optional): Whether to print a summary of the model's trainable
            parameters. Defaults to True.

    Returns:
        tuple[AutoModelForCausalLM, AutoTokenizer]: A tuple containing the loaded model,
            tokenizer, and PEFT configuration.
    """

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map=None,
        trust_remote_code=args.trust_remote_code,
        token=args.token,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
    )
    # tokenizer may miss some special tokens eg, pad_token in llama2
    smart_tokenizer_and_embedding_resize(
        model=model,
        tokenizer=tokenizer,
        smart_assign=True,
    )
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules if args.target_modules != 'auto' else find_target_modules(model),
        lora_dropout=args.lora_dropout,
        bias='none',
        task_type='CAUSAL_LM',
        use_dora=args.use_dora,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    wait_for_everyone()

    if print_summary and is_main_process():
        model.print_trainable_parameters()
        model_size_summary(model)

    return model, tokenizer, peft_config


def model_size_summary(model: nn.Module | AutoModel) -> None:
    """Print model summary

    Args:
        model (nn.Module | AutoModel)
    """
    def count_trainable_variables(module: nn.Module) -> int:
        return sum([np.prod(p.shape) for p in module.parameters() if p.requires_grad])

    def count_total_variables(module: nn.Module) -> int:
        return sum([np.prod(p.shape) for p in module.parameters()])

    total_params = count_total_variables(model)
    trainable_params = count_trainable_variables(model)
    name_max_len = max(len(m) for m in model._modules)
    n_modules = len(model._modules)
    horizontal_len = n_modules + name_max_len + 3 * 2 + len('Params (%)') + 4

    print('Model summary: ' + getattr(model, 'name_or_path', ''))
    print(f'{" "*len(str(n_modules))} | {"Name".ljust(name_max_len, " ")} | Params (ratio %)')
    print('-'*(horizontal_len))
    for n, m in enumerate(model._modules):
        params = count_total_variables(getattr(model, m))
        print(
            f'{str(n).ljust(len(str(n_modules)), " ")}'
            f' | {m.ljust(name_max_len, " ")}'
            f' | {params:>11,} ({params / total_params * 100:.2f}%)',
        )
    print('-'*(horizontal_len))
    print(f'Trainable params     {trainable_params:,}')
    print(f'Non-trainable params {total_params - trainable_params:,}')
    print(f'Total params         {total_params:,}')
    if hasattr(model, 'get_memory_footprint'):
        size_in_bytes = model.get_memory_footprint()
        print(f'Memory footprint     {size_in_bytes / 10**6:,.2f} MB (device={model.device})')
    else:
        # https://discuss.pytorch.org/t/finding-model-size/130275/2
        # https://pytorch.org/docs/stable/generated/torch.Tensor.element_size.html
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_in_mb = (param_size + buffer_size) / 10**6

        device = model.device if hasattr(model, 'device') else next(model.parameters()).device

        print(f'Memory footprint     {size_in_mb:,.2f} MB (device={device})')
    dtype = model.dtype if hasattr(model, 'dtype') else next(model.parameters()).dtype
    print(f'Model dtype          {dtype}')
