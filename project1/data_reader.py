from typing import List, Tuple

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

        self.n_source_tokens = len(set(w for l in self.source for w in l))
        self.n_target_tokens = len(set(w for l in self.target for w in l))

    def __getitem__(self, i) -> Tuple[Sentence, Sentence]:
        return self.source[i], self.target[i]

    def __len__(self) -> int:
        return len(self.source)
