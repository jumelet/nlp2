import numpy as np
from typing import Dict, Tuple, List
from collections import defaultdict
from tqdm import tqdm
import dill as pickle
from project1.data_reader import DataReader
from project1.aer import read_naacl_alignments, AERSufficientStatistics

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
                 gold_path_valid: str,
                 seeds=None) -> None:

        assert ibm_type.lower() in ['ibm1', 'ibm2'], \
            "Incorrect IBM model type, should be either 'IBM1' or 'IBM2'"
        self.ibm_type = ibm_type.lower()

        assert init_type.lower() in ['uniform', 'random', 'ibm1'], \
            "Incorrect init type, should be 'uniform', 'random', or 'IBM1'"
        self.init_type = init_type.lower()

        self.serialize_params = serialize_params
        self.train_data_reader = DataReader(source_path_train, target_path_train)
        self.valid_data_reader = DataReader(source_path_valid, target_path_valid)

        if self.init_type == 'uniform':
            init_ef_norm = 1 / self.train_data_reader.n_target_types
            self.f_given_e: Dict[Tuple[str, str], float] = defaultdict(lambda: init_ef_norm)
        elif self.init_type == 'ibm1':
            with open('translation_probs_IBM1.pickle', 'rb') as f:
                self.f_given_e = pickle.load(f)
        elif self.init_type == 'random':
            if not seeds:
                seeds = [1, 2, 3]
            self.seeds = seeds
            self.models = []

        if self.ibm_type == 'ibm2':
            init_align = 1 / max(map(len, self.train_data_reader.source))
            self.align_probs: Dict[int, float] = defaultdict(lambda: init_align)

        self.gold_links = read_naacl_alignments(gold_path_valid)


    def train(self, n_iter: int) -> None:
        if self.init_type != 'random':
            self._train(n_iter)
        else:
            for run_idx, seed in enumerate(self.seeds, start=1):
                print('Run {} with random initialisation.'.format(run_idx))
                np.random.seed(seed)
                self.f_given_e: Dict[Tuple[str, str], float] = defaultdict(lambda: np.random.random())

                _, aers = self._train(n_iter)
                self.models.append((np.copy(self.f_given_e), aers[-1]))

            # Compute aggregate model statistics
            _, models_aer = zip(*self.models)
            print('AER: mean {:.3f}  std: {:.3f}'.format(np.mean(models_aer), np.std(models_aer)))

            # Select the best model based on its AER
            best_model_idx = np.argmin(models_aer).item()
            best_aer = models_aer[best_model_idx]

            print('Best model: {}.  AER: {:.3f}'.format(best_model_idx+1, best_aer))

            # Reset probabilities according to best model
            self.f_given_e = self.models[best_model_idx][0]

        if self.serialize_params:
            with open('translation_probs_{}.pickle'.format(self.ibm_type), 'wb') as f:
                prob_dict = self.f_given_e
                pickle.dump(prob_dict, f)
            if self.ibm_type == 'IBM2':
               with open('alignment_probs_IBM2.pickle', 'wb') as af:
                   align_dict = self.align_probs
                   pickle.dump(align_dict, af)

    def _train(self, n_iter: int) -> Tuple[List[float], List[float]]:
        training_log_likelihoods = []
        aers = []

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

                log_ll_normalizer = np.log(1 / len_e) if self.ibm_type == 'ibm1' else 0

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
                        if self.ibm_type == 'ibm2':
                            jump = self.get_jump(e_pos, f_pos, len_e, len_f)
                            counts_align[jump] += delta

                    training_log_likelihood += np.log(log_likelihood) - log_ll_normalizer

            print('Training log-likelihood: {}'.format(training_log_likelihood))

            # Maximization
            for (we, wf), c in counts_ef.items():
                self.f_given_e[wf, we] = c / counts_e[we]

            if self.ibm_type == 'ibm2':
                norm_align_probs = np.sum(list(counts_align.values()))
                for x, c in counts_align.items():
                    self.align_probs[x] = c / norm_align_probs

            aer = self.validation(iteration)
            aers.append(aer)

        return (training_log_likelihoods, aers)



    def calc_ef_prob(self, wf: str, we: str, e_pos: int, f_pos: int, len_e: int, len_f: int):
        if self.ibm_type == 'ibm2':
            # TODO: add switch between jump/positional
            jump = self.get_jump(e_pos, f_pos, len_e, len_f)
            return self.f_given_e[wf, we] * self.align_probs[jump]

        return self.f_given_e[wf, we]

    def validation(self, iteration: int) -> float:
        print('Validation...')
        metric = AERSufficientStatistics()
        predictions = []

        for source, target in self.valid_data_reader.get_parallel_data():

            source = [NULL_TOKEN] + source

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
                link = (maxlink, 1 + f_pos)

                # Do not add links that align target words to the NULL token
                if maxlink != 0:
                    links.add(link)
            predictions.append(links)

        for gold, pred in zip(self.gold_links, predictions):
            metric.update(sure=gold[0], probable=gold[1], predicted=pred)

        aer = metric.aer()
        print(f'AER: {iteration} {aer:.3f}')

        return aer

    @staticmethod
    def get_jump(e_pos: int, f_pos: int, len_e: int, len_f: int) -> int:
        return e_pos - np.floor(f_pos * (len_e / len_f))
