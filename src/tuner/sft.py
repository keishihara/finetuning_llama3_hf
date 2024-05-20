import datetime
import time

import datasets
from transformers import (
    TrainingArguments,
    enable_full_determinism,
)

from tuner.configs.dataset_configs import DatasetConfig
from tuner.configs.training_args import ScriptArguments
from tuner.data import (
    DataCollator,
    prepare_dataset,
)
from tuner.modeling.trainer import SFTTrainer
from tuner.utils import (
    copy_demo_scripts,
    get_quantization_config_and_dtype,
    is_main_process,
    load_dataset_configs,
    load_model_and_tokenizer,
    print_on_rank_0,
    sanity_check_args,
    save_configs,
    setup_logging_dir,
    wait_for_everyone,
)

if not is_main_process():
    datasets.disable_progress_bar()


def train(
    args: ScriptArguments,
    dataset_configs_or_names: list[str | DatasetConfig],
) -> None:

    sanity_check_args(args)
    setup_logging_dir(args, prefix=args.prefix, add_hostname=False)
    copy_demo_scripts(args)

    if not isinstance(dataset_configs_or_names, list):
        dataset_configs_or_names = [dataset_configs_or_names]

    dataset_configs: list[DatasetConfig] = []
    for cfg_or_name in dataset_configs_or_names:
        if isinstance(cfg_or_name, str):
            # load preset datasets
            dataset_configs = [*dataset_configs, load_dataset_configs(cfg_or_name)[0]]
        elif isinstance(cfg_or_name, DatasetConfig):
            # append user-provided dataset config
            dataset_configs = [*dataset_configs, cfg_or_name]
        else:
            msg = f'Found invalid type instance in dataset_configs_or_names: {cfg_or_name}'
            raise TypeError(msg)
    save_configs(args.logging_dir, script_args=args, **{cfg.name: cfg for cfg in dataset_configs})

    if args.seed is not None:
        # For reproducible behavior during distributed training.
        # This will moderately reduce the training speed while conserving a bit more of VRAM.
        enable_full_determinism(args.seed)

    bnb_config, torch_dtype = get_quantization_config_and_dtype(args)
    model, tokenizer, peft_config = load_model_and_tokenizer(args, bnb_config, torch_dtype)
    save_configs(args.logging_dir, bnb_config=bnb_config, peft_config=peft_config)

    wait_for_everyone()

    ds = prepare_dataset(
        dataset_configs,
        tokenizer=tokenizer,
        model_max_length=args.model_max_length,
        split='train,eval',
    )
    train_ds, eval_ds = ds['train'], ds['eval']

    training_args = TrainingArguments(
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_drop_last=args.dataloader_drop_last,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        output_dir=args.output_dir,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        push_to_hub=args.push_to_hub,
        ddp_find_unused_parameters=False,
        optim=args.optim,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
    )
    save_configs(args.logging_dir, training_args=training_args)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollator(tokenizer),
    )
    wait_for_everyone()

    time_enter = time.time()
    model.config.use_cache = False
    trainer.train()
    model.config.use_cache = True
    time_elapsed = str(datetime.timedelta(seconds=time.time() - time_enter))
    print_on_rank_0(f'Training wall time: {time_elapsed}')
