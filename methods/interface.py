from abc import ABC, abstractmethod
from dataclasses import dataclass

from utils import log
from dataset import Candidate


# The class whose object is used to store the performance at each round
@dataclass
class Response:
    preds: list[str]
    n_hits: int = -1


class Strategy(ABC):
    """This is the interface that all strategies in this package will implement.
    It can be imported for type annotations."""

    @abstractmethod
    def get_preds(
        self,
        logger: log.InstanceLogger,
        mols: list[Candidate],
        n_rounds: int,
        sample_per_round: int,
    ):
        """Return the round wise performance of the strategy."""

    def get_hits_list(cls, item_list: list[Candidate]) -> list[str]:
        new_list = sorted(item_list, key=lambda x: x.score, reverse=True)
        num_hits = int(0.1 * len(item_list))
        hit_targets = [elem.name for elem in new_list[:num_hits]]
        return hit_targets
