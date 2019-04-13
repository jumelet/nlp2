from collections import defaultdict
from tqdm import tqdm

from .data_reader import DataReader

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

    def __init__(self, source_path, target_path):
        """
        :param source_path: str
            The path to the source corpus. Whitespace separates tokens, newline separates sequences (sentences).
        :param target_path: str
            The path to the target corpus. Whitespace separates tokens, newline separates sequences (sentences).
            The target corpus should have the same length of the source corpus.
        """
        self.data_reader = DataReader(source_path, target_path)
        self.translation_probs = defaultdict(lambda: EPSILON)


    def train(self, n_iterations):
        """
        Runs the Expectation Maximization algorithm to estimate translation probabilities.

        :param n_iterations: int
            The number of EM iterations over the parallel corpus.
        :return: None
        """
        for _ in range(n_iterations):

            translation_counts = defaultdict(float)  # How often do 'e' and 'f' form a translation pair?
            occurrence_counts = defaultdict(float)   # How often does 'e' appear in the corpus?

            for (source, target) in tqdm(self.data_reader.get_parallel_data(), total=len(self.data_reader)):

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

            for (s,t) in translation_counts:
                self.translation_probs[(t, s)] = translation_counts[(s, t)] / occurrence_counts[s]
