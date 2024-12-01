from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import Any


class Candidate:
    """
    Base class definition for a candidate
    """

    name: str
    score: float

    def __init__(self, cand_name, score):
        self.name = cand_name
        self.score = score

    def to_dict(self, dataset) -> dict[str, Any]:
        """Prepare the dictionary version of the data for memory store"""
        return {"name": self.name, "score": self.score, "explored": False}

    def conv_to_dict(self):
        """Prepare the dictionary version of the object data"""
        return {"name": self.name, "score": self.score}

    def __hash__(self):
        return hash(self.identifier)


class Dataset(ABC):
    """This is the interface that a dataset implements in order to be evaluated on."""

    def __init__(self, data_dir: str | PathLike):
        """Initialize the dataset object given the path to the data directory.

        This is usually just "./datasets/data_csv".
        """
        self.data_dir = Path(data_dir)

    @abstractmethod
    def _get_data(self) -> list[Candidate]:
        """Return the full list of candidates."""
