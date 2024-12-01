from dataset import Candidate
from utils import log
from .interface import Response, Strategy
import numpy as np
from retrieval import embedder_from_name
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qLogExpectedImprovement
import os
import torch
import copy


class GPR(Strategy):
    """This model chooses the top N genes from a gaussian process regression model"""

    def __init__(self, dataset: str, embedder: str):
        super().__init__()
        self.dataset = dataset
        self.embedder = embedder_from_name(embedder, dataset)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embeddings = None

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

    def optimize_model(self, x_train, y_train):
        self.gp_model = SingleTaskGP(
            x_train.to(self.device), y_train.to(self.device)
        ).to(self.device)
        mll = ExactMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model)
        fit_gpytorch_mll(mll)

    def acquire_new_points(self, y_best, available_mask, n_sample_per_round):
        self.gp_model.eval()
        qei = qLogExpectedImprovement(model=self.gp_model, best_f=y_best)
        candidate_set = self.embeds[available_mask == 1]
        candidate_ind = [
            i for i in range(available_mask.size(dim=0)) if available_mask[i] == 1
        ]

        selected_candidate_ind = []
        remaining_candidates = candidate_set.clone()

        # Loop to obtain batch by greedy sampling here
        for _ in range(n_sample_per_round):
            with torch.no_grad():
                qei_values = qei(
                    remaining_candidates.unsqueeze(0)
                )  # Add batch dimension
            best_idx = torch.argmax(qei_values)
            selected_candidate_ind.append(candidate_ind[best_idx])
            remaining_candidates = torch.cat(
                (remaining_candidates[:best_idx], remaining_candidates[best_idx + 1 :])
            )
            candidate_ind.pop(best_idx)

        return selected_candidate_ind

    def get_train_set(self, scores, available_mask):
        # available mask == 0 returns elements that have been explored previously
        x_train = self.embeds[available_mask == 0, :]
        y_train = scores[available_mask == 0]
        y_train = y_train.unsqueeze(-1).double()
        return x_train, y_train

    def get_hit_vector(self, hit_targets, mol_list):
        idx_list = [i for i in range(len(mol_list)) if mol_list[i] in hit_targets]
        theta_star = np.zeros(len(mol_list))
        theta_star[idx_list] = 1
        return theta_star

    def get_preds(
        self,
        logger: log.InstanceLogger,
        mols: list[Candidate],
        n_rounds: int,
        n_sample_per_round: int,
    ):
        hit_targets = self.get_hits_list(mols)
        mol_list = [mol.name for mol in mols]
        scores = torch.Tensor([mol.score for mol in mols]).to(self.device)
        hit_vector = self.get_hit_vector(hit_targets, mol_list)

        # This is to avoid computing the embeddings again for multiple runs as embeds are static
        if self.embeddings is None:
            self.embeds = self.load_embeddings(mols)
            self.embeds = self.embeds.astype(np.float64)
            self.embeds = torch.from_numpy(self.embeds).to(self.device)
            self.embeddings = copy.deepcopy(self.embeds)
        else:
            self.embeds = copy.deepcopy(self.embeddings)

        rd_wise_perf = []

        # Mask to maintain which candidates are already explored. 1 means candidate is unexplored
        available_mask = torch.ones(len(mols)).to(self.device)
        for t in range(n_rounds):
            if t == 0:
                chosen_action = np.random.choice(
                    len(mols), n_sample_per_round, replace=False
                ).tolist()
                available_mask[chosen_action] = 0
            else:
                x_train, y_train = self.get_train_set(scores, available_mask)
                # print("Got train set")
                self.optimize_model(x_train, y_train)
                # print('Model optimized')
                chosen_action = self.acquire_new_points(
                    y_train.max(), available_mask, n_sample_per_round
                )
                # print('Candidates acquired')

            available_mask[chosen_action] = 0

            rd_chosen_mols = np.array(mol_list)[chosen_action].tolist()
            rd_hits = np.sum(hit_vector[chosen_action])
            rd_wise_perf.append(Response(rd_chosen_mols, rd_hits))

            print(f"Round {t+1}: Discovered {rd_hits} hits")
            logger.write(f"Round {t+1}: Discovered {rd_hits} hits\n")

        return rd_wise_perf
