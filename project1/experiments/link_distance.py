import sys
import os
import pickle
import numpy as np
from collections import Counter
from pprint import pprint
import matplotlib.pyplot as plt

from project1.experiments.pos import read_alignments

class AlignmentModel(object):
    def __init__(self, model_path):
        sys.path.append(os.path.expanduser('..'))
        with open(model_path, 'rb') as file:
            self.probs = pickle.load(file)

    def get_links(self, source, target):
        links = set()
        for i, wt in enumerate(target, start=1):
            maxlink = np.argmax([self.probs[wt, ws] for ws in source])
            if maxlink != 0:
                links.add((maxlink, i))
        return list(links)


def link_distance_in_errors(model, en_corpus_path: str, fr_corpus_path: str, alignment_path: str):

    with open(en_corpus_path, 'r') as f_en, open(fr_corpus_path, 'r') as f_fr:
        en = [s.strip().split(' ') for s in f_en.readlines()]
        fr = [s.strip().split(' ') for s in f_fr.readlines()]

    en = {i: c for i, c in enumerate(en, start=1)}
    fr = {i: c for i, c in enumerate(fr, start=1)}

    all_a, sure_a, prob_a = read_alignments(alignment_path)

    errors_sure = Counter()
    errors_prob = Counter()
    errors_all = Counter()

    for idx in en.keys():
        e = ['NULL'] + en[idx]
        f = fr[idx]
        alignments = model.get_links(e, f)

        for a in alignments:

            if a in sure_a[idx]:
                continue
            if a in prob_a[idx]:
                continue

            # Else, if the predicted alignment is incorrect...

            ae, af = a
            pred_distance = abs(af - ae)

            for (gold_ae, gold_af) in sure_a[idx]:
                if gold_af == af:
                    gold_distance = abs(gold_af - gold_ae)
                    errors_sure[gold_distance - pred_distance] += 1
                    errors_all[gold_distance - pred_distance] += 1

            for (gold_ae, gold_af) in prob_a[idx]:
                if gold_af == af:
                    gold_distance = abs(gold_af - gold_ae)
                    errors_prob[gold_distance - pred_distance] += 1
                    errors_all[gold_distance - pred_distance] += 1

    return errors_all, errors_sure, errors_prob


def plot_distance_errors(errors):
    labels, values = zip(*sorted(errors.items(), key=lambda kv: kv[0]))
    indexes = np.arange(len(labels))
    width = 1

    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels)
    plt.show()


if __name__ == '__main__':
    m = AlignmentModel('../pickles/translation_probs_IBM1.pickle')
    errors_all, errors_s, errors_p = link_distance_in_errors(m,
                                     '../data/validation/dev.e',
                                     '../data/validation/dev.f',
                                     '../data/validation/dev.wa.nonullalign')


    plot_distance_errors(errors_all)