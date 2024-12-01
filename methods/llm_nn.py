import retrieval
from utils import log
import json
from .interface import Response, Strategy
from .prompts import _sys_prompt, _human_prompt
import pandas as pd


class LLMNN(Strategy):
    """This strategy prompts LLM to generate n centroids around which we select the
    nearest neighbours to form the prediction set."""

    mols: retrieval.DataStore
    k: int

    def __init__(self, model, docs: retrieval.DataStore, dataset: str):
        super().__init__()
        self.model = model
        self.mols = docs
        self.dataset = dataset
        self.round_wise_perf = []

    def get_dataset_specific_info_for_hum_prompt(self):
        with open(f"./dataset/task_prompts/{self.dataset}.json", "r") as f:
            data = json.load(f)

        return data["score_desc"], data["additional_info"]

    def store_round_performance(self, preds: list[str], targets: list[str]):
        hits = [elem for elem in preds if elem in targets]
        perf = Response(preds=preds, n_hits=len(hits))
        self.round_wise_perf.append(perf)
        return hits

    def build_feedback_str(self, rd_hits: list[str], expt_res):
        hits = []
        oth_res = []
        for obj in expt_res:
            if obj["name"] in rd_hits:
                hits.append(obj)
            else:
                oth_res.append(obj)

        hits_df = pd.DataFrame(hits)
        oth_res_df = pd.DataFrame(oth_res)
        feedback_str = (
            f"[HITS]\n{hits_df.to_string(index=False)}\n"
            f"[OTHER RESULTS]\n{oth_res_df.to_string(index=False)}"
        )

        return feedback_str

    def get_smiles(self, prompt):
        completion = self.model.generate(prompt)
        if "llama" in self.model.name:
            response = completion.split("<|end_header_id|>")[3]
            response = response[response.find("{") :].lstrip().rstrip()
        elif "mistral" in self.model.name:
            response = completion.split("[/INST]")[1].lstrip().rstrip()
            response = response.replace("'", "'")
        elif "qwen" in self.model.name:
            response = completion.split("<|im_start|>assistant\n")[1].lstrip().rstrip()
        smiles = json.loads(response)["SMILES"]
        return smiles

    def get_preds(
        self, logger: log.InstanceLogger, mols, n_rounds: int, sample_per_round: int
    ):
        """Return the round wise performance of the strategy."""
        pass

    def generate(
        self,
        logger: log.InstanceLogger,
        n_rounds: int,
        n_sample_per_round: int,
        n_centroids: int,
    ) -> list[Response]:
        all_objs = self.mols.return_all()
        hit_targets = self.get_hits_list(all_objs)
        hits = []
        expt_result = []
        rd = 1
        score_desc, add_info = self.get_dataset_specific_info_for_hum_prompt()
        max_retries = 3
        while rd <= n_rounds and max_retries > 0:
            logger.write(f"Round {rd}\n")
            if rd == 1:
                hum_prompt = _human_prompt.format(
                    score_description=score_desc,
                    additional_info=add_info,
                    num_candidates=len(all_objs),
                    batch_size=n_sample_per_round,
                    n_rounds=n_rounds,
                    rd=rd,
                    feedback="No feedback yet!",
                    n_centroids=n_centroids,
                )
            else:
                hum_prompt = _human_prompt.format(
                    score_description=score_desc,
                    additional_info=add_info,
                    num_candidates=len(all_objs),
                    batch_size=n_sample_per_round,
                    n_rounds=n_rounds,
                    rd=rd,
                    feedback=self.build_feedback_str(hits, expt_result),
                    n_centroids=n_centroids,
                )

            prompt = self.model.build_prompt(hum_prompt, _sys_prompt)
            if rd == 2:
                logger.log_prompt(prompt)
            try:
                search_smiles = self.get_smiles(prompt)
                max_retries = 3
            except:  # noqa: E722
                max_retries -= 1
                logger.write(f"Retrying again. Max Retries left: {max_retries}")
                continue

            print(f"Round {rd}: SMILES to search {search_smiles}")
            logger.write(f"Round {rd}: SMILES to search {search_smiles}\n")
            preds = []
            new_expt_result = []

            rem_batch_len = n_sample_per_round
            for i, smile in enumerate(search_smiles):
                if i == len(search_smiles) - 1:
                    num_to_choose = rem_batch_len
                else:
                    num_to_choose = int(n_sample_per_round / len(search_smiles))
                new_preds, temp_expt_result, _ = self.mols.search(
                    smile, num_to_choose, rd
                )

                preds.extend(new_preds)
                new_expt_result.extend(temp_expt_result)
                rem_batch_len -= len(new_preds)

            new_hits = self.store_round_performance(preds, hit_targets)
            print(f"Hits in round {rd}: {len(new_hits)}")
            logger.write(f"Hits in round {rd}: {len(new_hits)}\n")

            hits.extend(new_hits)
            expt_result.extend(new_expt_result)
            rd = rd + 1
        return self.round_wise_perf
