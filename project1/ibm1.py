from typing import Dict, Tuple
from collections import defaultdict
from tqdm import tqdm_notebook

from .data_reader import DataReader

NULL_TOKEN = 'NULL'


class IBM1:
    def __init__(self, source_path: str, target_path) -> None:
        self.data_reader = DataReader(source_path, target_path)

        init_ef_norm = 1 / (self.data_reader.n_source_tokens * self.data_reader.n_target_tokens)
        init_e_norm = 1 / self.data_reader.n_target_tokens
        self.probs_ef: Dict[Tuple[str, str], float] = defaultdict(lambda: init_ef_norm)
        self.probs_e: Dict[str, float] = defaultdict(lambda: init_e_norm)

    def train(self, n_iter: int):
        for s in range(n_iter):
            counts_ef = defaultdict(float)
            counts_e = defaultdict(float)

            for k in range(len(self.data_reader)):
                e, f = self.data_reader[k]
                e = [NULL_TOKEN] + e
                for we in e:
                    for wf in f:
                        delta = self.probs_ef[we, wf] / (self.probs_e[we])

                        if delta != 0:
                            counts_ef[we, wf] += delta
                            counts_e[we] += delta

            for (we, wf), c in counts_ef.items():
                self.probs_ef[we, wf] = c / counts_e[we]
                self.probs_e[we] += c / counts_e[we]
