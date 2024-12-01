import io
import os
import textwrap
from pathlib import Path
from typing import Iterable

from langchain_core.messages import BaseMessage


# for logging while performing BO
class InstanceLogger:
    _wrapped: io.TextIOWrapper

    def __init__(self, wrapped: io.TextIOWrapper):
        self._wrapped = wrapped

    @property
    def name(self) -> str:
        return self._wrapped.name

    def __enter__(self):
        self._wrapped = self._wrapped.__enter__()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._wrapped.__exit__(exc_type, exc_value, traceback)

    def write(self, s: str, /) -> int:
        return self._wrapped.write(s)

    def writelines(self, lines: Iterable[str], /):
        return self._wrapped.writelines(lines)

    def log_prompt(self, p: str | list[BaseMessage], kind: str = "PROMPT"):
        self._wrapped.write(f"BEGIN {kind}\n")
        self._wrapped.write(textwrap.indent(messages_to_string(p), "  "))
        self._wrapped.write(f"\nEND {kind}\n")


class HumanLogger:
    """This object logs results to a human-readable log folder, with separate files for each
    instance."""

    def __init__(self, parent_path: str | os.PathLike):
        self.path = Path(parent_path)

    def for_id(self, fname: str) -> InstanceLogger:
        """Return a logger for the specified example ID."""
        return InstanceLogger((self.path / f"{fname}.log").open("w"))


def messages_to_string(messages: str | list[BaseMessage]) -> str:
    """Convert what might be a list of chat messages into a loggable string."""
    if isinstance(messages, str):
        return messages

    def _format(msg: BaseMessage) -> str:
        if "\n" not in msg.content:
            return msg.__class__.__name__ + "(" + msg.content + ")"
        return (
            msg.__class__.__name__ + "(\n  " + msg.content.replace("\n", "\n  ") + "\n)"
        )

    return "\n".join(_format(msg) for msg in messages)
