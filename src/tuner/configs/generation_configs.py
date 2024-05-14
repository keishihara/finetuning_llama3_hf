from dataclasses import dataclass, field

import transformers


@dataclass
class GenerationConfig(transformers.GenerationConfig):
    do_sample: bool = field(default=True)
    max_new_tokens: int = field(default=1024)
    use_cache: bool = field(default=True)
    repetition_penalty: float = field(default=1.2)
