from sklearn.metrics import pairwise_distances
from dataset import Candidate
from utils import log
from .interface import Response, Strategy
import numpy as np
from retrieval import embedder_from_name
import copy
import os


class Coreset(Strategy):
    """This strategy focuses purely on diversity. It chooses the samples at farthest distance from
    their respective nearest point in the seen data points"""

    def __init__(self, dataset: str, embedder: str):
        super().__init__()
        self.dataset = dataset
        self.embedder = embedder_from_name(embedder, dataset)
        self.embeddings = None

    def furthest_first(self, X, X_set, n):
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)

        idxs = []

        for i in range(n):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        return idxs

    def load_embeddings(self, item_list):
        vectors = None
        cand_list = [elem.name for elem in item_list]
        cache_dir = f"./cache/{self.dataset}/{self.embedder.model_str}"
        if os.path.exists(f"{cache_dir}/embeds_batch0.npy"):
            num_files = len(os.listdir(cache_dir))
            for i in range(num_files):
                batch_vec = np.load(
                    f"{cache_dir}/embeds_batch{i}.npy", allow_pickle=True
                )
                if vectors is None:
                    vectors = batch_vec
                else:
                    vectors = np.vstack((vectors, batch_vec))
        else:
            vectors = np.array(self.embedder.embed(cand_list, feedback=None))

        return vectors

    def get_preds(
        self,
        logger: log.InstanceLogger,
        mols: list[Candidate],
        n_rounds: int,
        sample_per_round: int,
    ):
        print("Starting AL process now")
        hit_targets = self.get_hits_list(mols)

        # This is to avoid computing the embeddings again for multiple runs as embeds are static
        if self.embeddings is None:
            embeds = self.load_embeddings(mols)
            self.embeddings = copy.deepcopy(embeds)
        else:
            embeds = copy.deepcopy(self.embeddings)
        # print("Embeddings done")
        mols = np.array(mols)

        # Randomly shuffled every time
        rand_perm = np.random.permutation(len(mols))
        embeds = embeds[rand_perm]
        mols = mols[rand_perm]
        explored = np.array([])
        rd_wise_response = []

        for rd in range(n_rounds):
            print(f"Starting round {rd}")
            chosen_idxs = self.furthest_first(embeds, explored, sample_per_round)
            mask = np.full(
                len(embeds), True, dtype=bool
            )  # Creates a new mask for the remaining embeds
            mask[chosen_idxs] = False  # Mark the chosen idxs as unavailable now
            new_explored = embeds[~mask]
            if len(explored) == 0:
                explored = new_explored
            else:
                explored = np.vstack((explored, new_explored))

            preds = mols[~mask]

            n_hits = np.sum([1 for elem in preds if elem.name in hit_targets])
            rd_wise_response.append(Response(preds, n_hits))
            print(f"Round {rd}: Number of hits is {n_hits}")
            logger.write(f"Round {rd}: Number of hits is {n_hits}\n")

            # Filter out only the unselected embeds and mols here for next round
            embeds = embeds[mask]
            mols = mols[mask]

        return rd_wise_response
