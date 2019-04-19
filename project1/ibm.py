import numpy as np
from typing import Dict, Tuple
from collections import defaultdict
from tqdm import tqdm
import dill as pickle
from data_reader import DataReader
from aer import read_naacl_alignments, AERSufficientStatistics

NULL_TOKEN = 'NULL'


class IBM:
    def __init__(self,
                 ibm_type: str,
                 init_type: str,
                 serialize_params: bool,
                 source_path_train: str,
                 target_path_train: str,
                 source_path_valid: str,
                 target_path_valid: str,
                 gold_path_valid) -> None:
        assert ibm_type in ['IBM1', 'IBM2'], 'Incorrect IBM type, should be either IBM1 or IBM2'
        self.ibm_type = ibm_type

        assert init_type in ['uniform', 'ibm1'], 'Incorrect IBM type, should be either IBM1 or IBM2'

        self.serialize_params = serialize_params
        self.train_data_reader = DataReader(source_path_train, target_path_train)
        self.valid_data_reader = DataReader(source_path_valid, target_path_valid)

        init_ef_norm = 1 / self.train_data_reader.n_target_types

        if init_type == 'uniform':
            self.f_given_e: Dict[Tuple[str, str], float] = defaultdict(lambda: init_ef_norm)

        elif init_type == 'ibm1':
            with open('translation_probs_IBM1.pickle', 'rb') as f:
                self.f_given_e = pickle.load(f)

            self.f_given_e: Dict[Tuple[str, str], float] = defaultdict(lambda: init_ef_norm)

        if ibm_type == 'IBM2':
            init_align = 1 / max(map(len, self.train_data_reader.source))
            self.align_probs: Dict[int, float] = defaultdict(lambda: init_align)

        self.gold_links = read_naacl_alignments(gold_path_valid)

    def train(self, n_iter: int) -> None:
        for iteration in range(n_iter):
            counts_ef = defaultdict(float)
            counts_e = defaultdict(float)
            counts_align = defaultdict(int)
            training_log_likelihood = 0

            # Expectation
            for (e, f) in tqdm(self.train_data_reader.get_parallel_data(),
                               total=len(self.train_data_reader)):

                e = [NULL_TOKEN] + e
                len_e = len(e)
                len_f = len(f)

                log_ll_normalizer = np.log(1 / len_e) if self.ibm_type == 'IBM1' else 0

                for f_pos, wf in enumerate(f):
                    delta_normalizer = 0
                    log_likelihood = 0
                    for e_pos, we in enumerate(e):
                        ef_prob = self.calc_ef_prob(wf, we, e_pos, f_pos, len_e, len_f)
                        delta_normalizer += ef_prob

                    for e_pos, we in enumerate(e):
                        ef_prob = self.calc_ef_prob(wf, we, e_pos, f_pos, len_e, len_f)
                        delta = ef_prob / delta_normalizer
                        log_likelihood += ef_prob

                        counts_ef[we, wf] += delta
                        counts_e[we] += delta
                        if self.ibm_type == 'IBM2':
                            jump = self.get_jump(e_pos, f_pos, len_e, len_f)
                            counts_align[jump] += delta

                    training_log_likelihood += np.log(log_likelihood) - log_ll_normalizer

            print('Training log-likelihood: {}'.format(training_log_likelihood))

            # Maximization
            for (we, wf), c in counts_ef.items():
                self.f_given_e[wf, we] = c / counts_e[we]

            if self.ibm_type == 'IBM2':
                norm_align_probs = np.sum(list(counts_align.values()))
                for x, c in counts_align.items():
                    self.align_probs[x] = c / norm_align_probs

            self.validation(iteration)

        if self.serialize_params:
            with open('translation_probs_{}.pickle'.format(self.ibm_type), 'wb') as f:
                prob_dict = self.f_given_e
                pickle.dump(prob_dict, f)

    def calc_ef_prob(self, wf: str, we: str, e_pos: int, f_pos: int, len_e: int, len_f: int):
        if self.ibm_type == 'IBM2':
            # TODO: add switch between jump/positional
            jump = self.get_jump(e_pos, f_pos, len_e, len_f)
            return self.f_given_e[wf, we] * self.align_probs[jump]

        return self.f_given_e[wf, we]

    def validation(self, iteration: int) -> None:
        print('Validation...')
        metric = AERSufficientStatistics()
        predictions = []

        for source, target in self.valid_data_reader.get_parallel_data():

            len_e = len(source)
            len_f = len(target)
            links = set()
            for f_pos, wf in enumerate(target):
                maxlink = np.argmax(
                    [
                        self.calc_ef_prob(wf, we, e_pos, f_pos, len_e, len_f)
                        for (e_pos, we) in enumerate(source)
                    ]
                )
                link = (1 + maxlink, 1 + f_pos)
                links.add(link)
            predictions.append(links)

        for gold, pred in zip(self.gold_links, predictions):
            metric.update(sure=gold[0], probable=gold[1], predicted=pred)

        aer = metric.aer()
        print(f'AER: {iteration} {aer:.3f}')

    @staticmethod
    def get_jump(e_pos: int, f_pos: int, len_e: int, len_f: int) -> int:
        return e_pos - np.floor(f_pos * (len_e / len_f))
