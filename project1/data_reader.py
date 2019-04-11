from typing import List, Tuple

from .w2i import W2I

Sentence = List[str]
Corpus = List[Sentence]


class DataReader:
    def __init__(self, source_path: str, target_path) -> None:
        with open(source_path, 'r') as f1:
            self.source: Corpus = [l.split(' ') for l in f1.read().split('\n')]
        with open(target_path, 'r') as f2:
            self.target: Corpus = [l.split(' ') for l in f2.read().split('\n')]

        assert len(self.source) == len(self.target), \
            'Parallel corpus not of equal size!'

        self.source_w2i, self.target_w2i = self._create_vocabs()

    def __getitem__(self, i) -> Tuple[Sentence, Sentence]:
        return self.source[i], self.target[i]

    def __len__(self) -> int:
        return len(self.source)

    def _create_vocabs(self):
        source_unique_w = set(w for l in self.source for w in l)
        target_unique_w = set(w for l in self.target for w in l)

        source_w2i = W2I({k+1: w for k, w in enumerate(source_unique_w)})
        source_w2i['NULL'] = 0

        target_w2i = W2I({k: w for k, w in enumerate(target_unique_w)})

        return source_w2i, target_w2i
