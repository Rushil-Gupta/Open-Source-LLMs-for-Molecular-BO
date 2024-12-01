from .base import Dataset, Candidate
from os import PathLike
from pathlib import Path
import pandas as pd


class Molecules(Dataset):
    # The name of the column in the data csv to extract as score
    _score_map = {
        "esol": "measured log solubility in mols per litre",
        "freesolv": "expt",
        "nextgen": "IE",
    }

    def __init__(self, data_csv_dir: str | PathLike, data_name: str):
        """Initialize the dataset object given the path to the data directory.

        This is usually just "./datasets/data_csv".
        """
        super().__init__(data_csv_dir)
        self.dataset = data_name

    def _get_data(self) -> list[Candidate]:
        mol_data_path = Path(self.data_dir) / f"{self.dataset}_dataset.csv"
        data_source = pd.read_csv(mol_data_path, header=0)
        score_field = self._score_map[self.dataset]
        mol_list = []
        for index, row in data_source.iterrows():
            mol_list.append(Candidate(row["SMILES"], row[score_field]))

        print(f"We have {len(mol_list)} number of candidates here")

        return mol_list
