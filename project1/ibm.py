from collections import defaultdict
from tqdm import tqdm
import numpy as np

from .data_reader import DataReader
from .aer import read_naacl_alignments, AERSufficientStatistics

NULL_TOKEN = '[NULL]'
EPSILON = 1e-12


class IBM1(object):
    """
    IBM Model 1.

    Attributes
    ----------
    translation_probs: Dict[Tuple[str, str] -> float]
        A matrix of probabilities indicating the likelihood of all possible translation pairs.

    Methods
    -------
    train(n_iterations)
        Runs the Expectation Maximization algorithm to estimate translation probabilities.
    """

    def __init__(self, source_path, target_path, source_path_valid, target_path_valid, gold_path_valid):
        """
        :param source_path: str
            The path to the source corpus. Whitespace separates tokens, newline separates sequences (sentences).
        :param target_path: str
            The path to the target corpus. Whitespace separates tokens, newline separates sequences (sentences).
            The target corpus should have the same length of the source corpus.
        """
        self.training_data = DataReader(source_path, target_path)
        self.validation_data = DataReader(source_path_valid, target_path_valid)
        self.gold_links = read_naacl_alignments(gold_path_valid)

        self.translation_probs = defaultdict(lambda: EPSILON)


    def train(self, n_iterations):
        """
        Runs the Expectation Maximization algorithm to estimate translation probabilities.

        :param n_iterations: int
            The number of EM iterations over the parallel corpus.
        :return: None
        """
        print('Start training...')

        # Collect convergence statistics
        training_log_lik_per_epoch = []
        validation_aer_per_epoch = []

        for iter in range(n_iterations):

            print('Epoch {}.'.format(iter))

            # EXPECTATION MAXIMISATION
            translation_counts = defaultdict(float)  # How often do 'e' and 'f' form a translation pair?
            occurrence_counts = defaultdict(float)   # How often does 'e' appear in the corpus?
            training_log_likelihood = 0

            for (source, target) in tqdm(self.training_data.get_parallel_data(), total=len(self.training_data)):

                # Add NULL token to naively cope with fertility
                source = [NULL_TOKEN] + source

                for t in target:

                    # First compute normalisation constant for this target word 'f'.
                    # It is necessary to update all counts involving 'f'.
                    delta_denominator = 0
                    for s in source:
                        delta_denominator += self.translation_probs[(t, s)]

                    # Now update the counts.
                    for s in source:
                        delta = self.translation_probs[(t, s)] / delta_denominator

                        translation_counts[(s, t)] += delta
                        occurrence_counts[s] += delta

                    training_log_likelihood += np.log(np.max([self.translation_probs[(t, s)] for s in source]))

            print('Training log-likelihood: {}'.format(training_log_likelihood))
            training_log_lik_per_epoch.append(training_log_likelihood)

            for (s,t) in translation_counts:
                self.translation_probs[(t, s)] = translation_counts[(s, t)] / occurrence_counts[s]


            # DECODING
            print('Validation... ')
            metric = AERSufficientStatistics()
            predictions = []

            for (source, target) in tqdm(self.validation_data.get_parallel_data(), total=len(self.validation_data)):

                source = [NULL_TOKEN] + source

                links = set()
                for i, t in enumerate(target, start=1):
                    link = (
                        1 + np.argmax([self.translation_probs[(t,s)] for s in source]),
                        i
                    )
                    links.add(link)
                predictions.append(links)

            for gold, pred in zip(self.gold_links, predictions):
                metric.update(sure=gold[0], probable=gold[1], predicted=pred)

            aer = metric.aer()
            print('AER: {}'.format(iter, aer))
            validation_aer_per_epoch.append(aer)

        return training_log_lik_per_epoch, validation_aer_per_epoch


class IBM2(object):
    """
    IBM Model 2.

    Attributes
    ----------
    translation_probs: Dict[Tuple[str, str] -> float]
        A matrix of probabilities indicating the likelihood of all possible translation pairs.

    alignment_probs: Dict[Tuple[int, int, int, int] -> float]
        Alignment probabilities q(j|i,l,m) indicating the likelihood of a target word at position j to be aligned with
        source word i, given a source and a target sentence of length l and m respectively.

    Methods
    -------
    train(n_iterations)
        Runs the Expectation Maximization algorithm to estimate translation and alignment probabilities.
    """

    def __init__(self, source_path, target_path, source_path_valid, target_path_valid, gold_path_valid, alignment='positional'):
        """
        :param source_path: str
            The path to the source corpus. Whitespace separates tokens, newline separates sequences (sentences).
        :param target_path: str
            The path to the target corpus. Whitespace separates tokens, newline separates sequences (sentences).
            The target corpus should have the same length of the source corpus.
        :param alignment: str
            Positional alignment ('positional') or Vogel et al.'s jump distribution ('jump').
        """
        if alignment not in ['positional', 'jump']:
            raise ValueError("Alignment can be 'positional' or according to a 'jump' distribution")
        self.alignment = alignment

        self.alignment_probs = defaultdict(lambda: EPSILON)
        self.translation_probs = defaultdict(lambda: EPSILON)

        self.training_data = DataReader(source_path, target_path)
        self.validation_data = DataReader(source_path_valid, target_path_valid)

        self.gold_links = read_naacl_alignments(gold_path_valid)


    def train(self, n_iterations):
        if self.alignment == 'positional':
            return self._train_positional()
        else:
            raise NotImplementedError('Jump distribution not yet implemented!')


    def _train_positional(self, n_iterations):
        """
        Runs the Expectation Maximization algorithm to estimate translation and alignment probabilities.

        :param n_iterations: int
            The number of EM iterations over the parallel corpus.
        :return: None
        """
        print('Start training...')

        # Collect convergence statistics
        training_log_lik_per_epoch = []
        validation_aer_per_epoch = []

        for iter in range(n_iterations):

            print('Epoch {}.'.format(iter))

            # EXPECTATION MAXIMISATION
            translation_counts = defaultdict(float)  # How often do 'e' and 'f' form a translation pair?
            occurrence_counts = defaultdict(float)   # How often does 'e' appear in the corpus?
            training_log_likelihood = 0

            # How often does the ith target word align with the jth source word...
            # given a target and a source sentence of length m and l respectively?
            positional_alignment_counts = defaultdict(float)

            # How often does an l-long source sentence align with an m-long target sentence?
            length_alignment_counts = defaultdict(float)

            for (source, target) in tqdm(self.training_data.get_parallel_data(), total=len(self.training_data)):

                # Add NULL token to naively cope with fertility
                source = [NULL_TOKEN] + source

                l = len(source)
                m = len(target)

                for i, t in enumerate(target, start=1):

                    # First compute normalisation constant for this target word 'f'.
                    # It is necessary to update all counts involving 'f'.
                    delta_denominator = 0
                    for (j, s) in enumerate(source, start=1):
                        delta_denominator += self.alignment_probs[(j, i, l, m)] * self.translation_probs[(t, s)]

                    # Now update the counts.
                    for (j, s) in enumerate(source, start=1):
                        delta = self.translation_probs[(t, s)] / delta_denominator

                        translation_counts[(s, t)] += delta
                        occurrence_counts[s] += delta

                        positional_alignment_counts[(j, i, l, m)] += delta
                        length_alignment_counts[(i, l, m)] += delta

                    training_log_likelihood += \
                        np.log(
                            np.max(
                                [self.translation_probs[(t, s)] * self.alignment_probs[(j, i, l, m)]
                                 for (j, s)
                                 in enumerate(source, start=1)])
                        )

            print('Training log-likelihood: {}'.format(training_log_likelihood))
            training_log_lik_per_epoch.append(training_log_likelihood)

            for (s,t) in translation_counts:
                self.translation_probs[(t, s)] = translation_counts[(s, t)] / occurrence_counts[s]

            for (j, i, l, m) in positional_alignment_counts:
                self.alignment_probs[(j, i, l, m)] = \
                    positional_alignment_counts[(j, i, l, m)] / length_alignment_counts[(i, l, m)]

            # DECODING
            print('Validation...')
            metric = AERSufficientStatistics()
            predictions = []

            for (source, target) in tqdm(self.validation_data.get_parallel_data(), total=len(self.validation_data)):

                source = [NULL_TOKEN] + source

                l = len(source)
                m = len(target)
                links = set()
                for i, t in enumerate(target, start=1):
                    link = (
                        1 + np.argmax(
                            [self.translation_probs[(t, s)] * self.alignment_probs[(j, i, l, m)]
                             for (j, s)
                             in enumerate(source, start=1)]),
                        i
                    )
                    links.add(link)
                predictions.append(links)

            for gold, pred in zip(self.gold_links, predictions):
                metric.update(sure=gold[0], probable=gold[1], predicted=pred)

            aer = metric.aer()
            print('AER: {}'.format(iter, aer))
            validation_aer_per_epoch.append(aer)

        return training_log_lik_per_epoch, validation_aer_per_epoch


