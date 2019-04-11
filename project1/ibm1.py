import numpy as np

from .data_reader import DataReader


class IBM1:
    def __init__(self, source_path: str, target_path) -> None:
        self.data_reader = DataReader(source_path, target_path)

        self.counts = self._reset_counts()
        self.probs = self._reset_counts()

    def train(self, n_iter: int):
        for s in range(n_iter):
            self.counts = self._reset_counts()
            for k in range(len(self.data_reader)):
                e, f = self.data_reader[k]
                for i in range(1, len(f)):
                    for j in range(len(e)):
                        delta = self.props

    def _reset_counts(self) -> np.ndarray:
        n_source_tokens = len(self.data_reader.source_w2i)
        n_target_tokens = len(self.data_reader.target_w2i)

        return np.zeros((n_source_tokens, n_target_tokens))
