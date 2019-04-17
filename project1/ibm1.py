import numpy as np
from typing import Dict, Tuple
from collections import defaultdict
from tqdm import tqdm

from .data_reader import DataReader
from .aer import read_naacl_alignments, AERSufficientStatistics

NULL_TOKEN = 'NULL'


class IBM1:
    def __init__(self,
                 source_path_train: str,
                 target_path_train,
                 source_path_valid,
                 target_path_valid,
                 gold_path_valid) -> None:

        self.train_data_reader = DataReader(source_path_train, target_path_train)
        self.valid_data_reader = DataReader(source_path_valid, target_path_valid)

        init_ef_norm = 1 / self.train_data_reader.n_target_types
        self.f_given_e: Dict[Tuple[str, str], float] = defaultdict(lambda: init_ef_norm)

        self.gold_links = read_naacl_alignments(gold_path_valid)

    def train(self, n_iter: int):
        for iteration in range(n_iter):
            counts_ef = defaultdict(float)
            counts_e = defaultdict(float)
            training_log_likelihood = 0

            # Expectation
            for (e, f) in tqdm(self.train_data_reader.get_parallel_data(),
                               total=len(self.train_data_reader)):

                e = [NULL_TOKEN] + e
                len_e = len(e)

                for wf in f:
                    delta_normalizer = 0
                    for we in e:
                        delta_normalizer += self.f_given_e[wf, we]

                    for we in e:
                        delta = self.f_given_e[wf, we] / delta_normalizer
                        counts_ef[we, wf] += delta
                        counts_e[we] += delta

                    training_log_likelihood += np.log(np.sum([self.f_given_e[wf, we] for we in e])) - np.log(1 / len_e)

            print('Training log-likelihood: {}'.format(training_log_likelihood))

            # Maximization
            for (we, wf), c in counts_ef.items():
                self.f_given_e[wf, we] = c / counts_e[we]

            self.validation(iteration)

    def validation(self, iteration: int):
        print('Validation...')
        metric = AERSufficientStatistics()
        predictions = []

        for source, target in tqdm(self.valid_data_reader.get_parallel_data(),
                                   total=len(self.valid_data_reader)):

            links = set()
            for i, wf in enumerate(target):
                link = (
                    1 + np.argmax([self.f_given_e[wf, we] for we in source]),
                    1 + i
                )
                links.add(link)
            predictions.append(links)

        for gold, pred in zip(self.gold_links, predictions):
            metric.update(sure=gold[0], probable=gold[1], predicted=pred)

        aer = metric.aer()
        print(f'AER: {iteration} {aer:.3f}')