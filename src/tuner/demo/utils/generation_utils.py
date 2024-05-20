import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
)


@torch.inference_mode()
def generate(
    messages: list[dict],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    streamer: TextStreamer | None = None,
    generation_config: dict | None = None,
) -> str:
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
