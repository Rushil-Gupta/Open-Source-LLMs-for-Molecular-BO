from dataset import Candidate
from utils import log
from .interface import Response, Strategy
import numpy as np


class Random(Strategy):
    """This model chooses the molecules randomly for prediction"""

    def __init__(self, dataset: str):
        super().__init__()
        self.dataset = dataset

    def get_preds(
        self,
        logger: log.InstanceLogger,
        mols: list[Candidate],
        n_rounds: int,
        sample_per_round: int,
    ):
        hit_targets = self.get_hits_list(mols)

        idx_list = np.arange(len(mols))
        rd_wise_response = []

        for rd in range(n_rounds):
            chosen_idxs = np.random.choice(
                idx_list, size=sample_per_round, replace=False
            )
            preds = np.array(mols)[chosen_idxs]
            n_hits = np.sum([1 for elem in preds if elem.name in hit_targets])
            rd_wise_response.append(Response(preds, n_hits))
            logger.write(f"Hits in round {rd+1}: {n_hits}\n")
            idx_list = np.array(list(set(idx_list.tolist()) - set(chosen_idxs)))

        return rd_wise_response
