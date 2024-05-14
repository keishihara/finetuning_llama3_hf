import textwrap
from abc import ABC, abstractmethod
from collections.abc import Iterable

from tuner._typing import Chat


class FormatterMixin(ABC):
    @abstractmethod
    def format(self, **inputs) -> Chat:
        """Returns text data in chat messages format, like below:
        [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
            {"role": "user", "content": "I'd like to show off how chat templating works!"},
            ...
        ]
        """

    def batch_format(self, batch_inputs: Iterable[dict]) -> list[Chat]:
        """
        Applies format method above to batch_inputs and
        returns a batch of formatted prompt strings.
        """
        return [self.format(**inputs) for inputs in batch_inputs]

    def __call__(self, *args, **kwargs) -> Chat | list[Chat]:
        """
        Without using keyword arguments when calling, it assumes a single batch of
        inputs (Iterable) incoming.
        """
        if args and not kwargs:
            batch_inputs = args[0]
            if not isinstance(batch_inputs, Iterable):
                raise TypeError
            return self.batch_format(batch_inputs)
        return self.format(**kwargs)

    def cutout_last_assistant_message(self, text: str, startswith: str) -> str:
        response_start_index = text.rfind(startswith) + len(startswith)
        return text[response_start_index:]

    @staticmethod
    def dedent(text: str | list[str]) -> str:
        """Removes indentation from input text(s)."""
        if isinstance(text, list):
            return [textwrap.dedent(t) for t in text]
        return textwrap.dedent(text)
