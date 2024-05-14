from dataclasses import dataclass, field


@dataclass
class InferenceArguments:
    model_name_or_path: str = field(default='meta-llama/Meta-Llama-3-8B-Instruct')
    out_path: str = field(default='auto')
    dataset_config_names: list[str] = None
    split: str = field(default='test')
    eval_size: int = field(default=0.1)
    load_in_4bit: bool = field(default=False)
    load_in_8bit: bool = field(default=True)
    model_max_length: int = field(default=2048)
    per_device_batch_size: int = field(default=1)
    dataloader_num_workers: int = field(default=8)
    use_flash_attention_2: bool = field(default=True)
