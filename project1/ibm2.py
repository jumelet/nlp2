import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple
from collections import defaultdict

from project1.data_reader import DataReader
from .aer import read_naacl_alignments, AERSufficientStatistics

NULL_TOKEN = 'NULL'


class IBM2:
    def __init__(self,
                 longest_s: int,
                 source_path_train: str,
                 target_path_train,
                 source_path_valid,
                 target_path_valid,
                 gold_path_valid) -> None:

        self.train_data_reader = DataReader(source_path_train, target_path_train)
        self.valid_data_reader = DataReader(source_path_valid, target_path_valid)

        init_ef_norm = 1 / (self.train_data_reader.n_source_types * self.train_data_reader.n_target_types)
        init_align = 1 / (longest_s)

        self.probs_ef: Dict[Tuple[str, str], float] = defaultdict(lambda: init_ef_norm)
        self.align_probs: Dict[int, float] = defaultdict(lambda: init_align)

        self.gold_links = read_naacl_alignments(gold_path_valid)

    def get_jump(self, e_pos, f_pos, len_e, len_f):
        return e_pos - np.floor(f_pos * (len_e / len_f))

    def train(self, n_iter: int):
        for iter in range(n_iter):

            counts_ef = defaultdict(float)
            counts_e = defaultdict(float)
            counts_align = defaultdict(int)
            training_log_likelihood = 0

            for (e, f) in tqdm(self.train_data_reader.get_parallel_data(), total=len(self.train_data_reader)):
                e = [NULL_TOKEN] + e

                len_e = len(e)
                len_f = len(f)

                e_normalizer = defaultdict(float)

                for e_pos, we in enumerate(e):
                    for f_pos, wf in enumerate(f):
                        jump = self.get_jump(e_pos, f_pos, len_e, len_f)
                        e_normalizer[we] += self.probs_ef[we, wf] * self.align_probs[jump]

                for f_pos, wf in enumerate(f):
                    for e_pos, we in enumerate(e):

                        jump = self.get_jump(e_pos, f_pos, len_e, len_f)

                        delta = (self.probs_ef[we, wf] * self.align_probs[jump]) / e_normalizer[we]

                        counts_ef[we, wf] += delta
                        counts_e[wf] += delta
                        counts_align[jump] += delta

                    training_log_likelihood += \
                        np.log(np.sum([self.probs_ef[(wf, we)] for we in e])) - np.log(1 / len_e)

            print('Training log-likelihood: {}'.format(training_log_likelihood))

            for (we, wf), c in counts_ef.items():
                self.probs_ef[we, wf] = c / counts_e[wf]

            norm_align_probs = np.sum(list(counts_align.values()))
            for x, c in counts_align.items():
                self.align_probs[x] = c / norm_align_probs

            self.validation(iter)

    def validation(self, iter):
        print('Validation...')
        metric = AERSufficientStatistics()
        predictions = []

        for (source, target) in tqdm(self.valid_data_reader.get_parallel_data(), total=len(self.valid_data_reader)):

            source = [NULL_TOKEN] + source

            l = len(source)
            m = len(target)
            links = set()
            for i, t in enumerate(target):
                link = (
                    1 + np.argmax([self.probs_ef[(t, s)] for (j, s) in enumerate(source)]),
                    1 + i
                )
                links.add(link)
            predictions.append(links)

        for gold, pred in zip(self.gold_links, predictions):
            metric.update(sure=gold[0], probable=gold[1], predicted=pred)

        aer = metric.aer()
        print(f'AER: {iter} {aer:.3f}')