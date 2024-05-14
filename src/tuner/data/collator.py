from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from tuner._constants import IGNORE_INDEX


class DataCollator:
    def __init__(self, tokenizer: AutoTokenizer, ignore_index: int = IGNORE_INDEX) -> None:
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

    def __call__(self, examples: list[dict]) -> dict:
        """
        Data collator for causal modeling in general.

        Return:
            dict[str, torch.Tensor]: a dict of batch of Tensors
        """
        batch_input = [example['input_ids'] for example in examples]
        batch_label = [example['labels'] for example in examples]
        input_ids = pad_sequence(
            batch_input, batch_first=True, padding_value=self.tokenizer.pad_token_id,
        )
        labels = pad_sequence(
            batch_label, batch_first=True, padding_value=self.ignore_index,
        )
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }
