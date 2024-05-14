from dataclasses import dataclass, field

from tuner import LOG_DIR


@dataclass
class ScriptArguments:
    do_train: bool = field(default=False, metadata={'help': 'This argument is not directly used by Trainer.'})
    do_eval: bool = field(default=False, metadata={'help': 'This argument is not directly used by Trainer.'})
    do_predict: bool = field(default=False, metadata={'help': 'This argument is not directly used by Trainer.'})
    prefix: str = field(default='auto', metadata={'help': 'The string to be prefixed to the log directory. '
        'By setting `auto`, model_name_or_path will be used as the log directory name.'})
    # model
    model_name_or_path: str = field(default='meta-llama/Meta-Llama-3-8B-Instruct')
    model_max_length: int = field(default=8192, metadata={
        'help': 'Maximum sequence length. Sequences will be right padded by default (and possibly truncated).'})
    load_in_8bit: bool = field(default=False, metadata={'help': 'Load the model in 8 bits precision'})
    load_in_4bit: bool = field(default=False, metadata={'help': 'Load the model in 4 bits precision'})
    low_cpu_mem_usage: bool = field(default=True)
    # params
    dataloader_num_workers: int = field(default=8, metadata={'help': 'Number of data loader workers'})
    dataloader_drop_last: bool = field(default=False)
    # training args
    per_device_train_batch_size: int = field(default=1, metadata={'help': 'The batch size'})
    per_device_eval_batch_size: int = field(default=1, metadata={'help': 'The batch size'})
    optim: str = field(default='adamw_torch')
    learning_rate: float = field(default=2e-4, metadata={'help': 'The learning rate'})
    warmup_ratio: float = field(default=0.1, metadata={
        'help': 'Ratio of total training steps used for a linear warmup from 0 to learning_rate.'})
    gradient_accumulation_steps: int = field(default=1, metadata={
        'help': 'The number of gradient accumulation steps'})
    fp16: bool = field(default=False, metadata={'help': 'Enable float16 precision training.'})
    bf16: bool = field(default=False, metadata={'help': 'Enable bfloat16 precision training.'})
    seed: int = field(default=None, metadata={'help': 'the seed'})
    num_train_epochs: int = field(default=1, metadata={'help': 'Number of training epochs'})
    evaluation_strategy: str = field(default='epoch', metadata={
        'help': 'The evaluation strategy to adopt during training'})
    eval_steps: int = field(default=60)
    output_dir: str = field(default=str(LOG_DIR))
    save_strategy: str = field(default='epoch', metadata={
        'help': ' The checkpoint save strategy to adopt during training'})
    save_steps: int = field(default=60, metadata={'help': 'This has no effect if `save_strategy=epochs`.'})
    save_total_limit: int = field(default=10, metadata={'help': 'Total checkpoints to be written.'})
    logging_strategy: str = field(default='steps', metadata={'help': 'the logging strategy'})
    logging_steps: int = field(default=10, metadata={'help': 'the number of logging steps'})
    token: bool = field(default=True, metadata={'help': 'Use HF auth token to access the model'})
    trust_remote_code: bool = field(default=True)
    push_to_hub: bool = field(default=False)
    report_to: str = field(default='all')
    load_best_model_at_end: bool = field(default=False, metadata={
        'help': 'Enable this to save the best model at end (this has no effect currently)'})
    # peft
    lora_r: int = field(default=8, metadata={'help': 'the r parameter of the LoRA adapters'})
    lora_alpha: int = field(default=16, metadata={'help':
        'the alpha parameter of the LoRA adapters, which is recommeded to be set at double the value of lora r.'})
    target_modules: list[str] = field(default_factory=lambda:
        ['q_proj', 'up_proj', 'o_proj', 'k_proj', 'down_proj', 'gate_proj', 'v_proj'])
    lora_dropout: float = field(default=0.1)
    use_dora: bool = field(default=False)
    # flash attention 2
    attn_implementation: str = field(default='flash_attention_2')
