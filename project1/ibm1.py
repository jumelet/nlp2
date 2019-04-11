import numpy as np
from scipy.sparse import lil_matrix
from sklearn.preprocessing import normalize

from .data_reader import DataReader

NULL_TOKEN = 'NULL'


class IBM1:
    def __init__(self, source_path: str, target_path) -> None:
        self.data_reader = DataReader(source_path, target_path)

        self.probs: lil_matrix = self._reset_counts()

    def train(self, n_iter: int):
        for s in range(n_iter):
            counts = self._reset_counts()

            for k in range(len(self.data_reader)):
                e, f = self.data_reader[k]
                e = [NULL_TOKEN] + e
                for we in e:
                    for wf in f:
                        we_i = self.data_reader.source_w2i[we]
                        wf_i = self.data_reader.target_w2i[wf]

                        delta = self.probs[we_i, wf_i] / np.sum(self.probs[:, wf_i])

                        counts[we_i, wf_i] += delta

            self.probs = normalize(counts, norm='l1', axis=1)

    def _reset_counts(self) -> lil_matrix:
        n_source_tokens = len(self.data_reader.source_w2i)
        n_target_tokens = len(self.data_reader.target_w2i)

        return lil_matrix((n_source_tokens, n_target_tokens))+1
