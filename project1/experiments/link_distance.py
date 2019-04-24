import sys
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from pprint import pprint

from project1.experiments.pos import read_alignments


class AlignmentModel(object):
    def __init__(self, translation_model_path: str, alignment_model_path=None) -> None:
        sys.path.append(os.path.expanduser('..'))

        with open(translation_model_path, 'rb') as file:
            self.translation_probs = pickle.load(file)

        self.alignment_probs = None
        if alignment_model_path:
            with open(alignment_model_path, 'rb') as file:
                self.alignment_probs = pickle.load(file)

    @staticmethod
    def get_jump(e_pos: int, f_pos: int, len_e: int, len_f: int) -> int:
        return e_pos - np.floor(f_pos * (len_e / len_f))

    def get_links(self, source, target):
        len_s, len_t = len(source), len(target)
        links = set()
        for i, wt in enumerate(target, start=1):
            if self.alignment_probs:
                maxlink = np.argmax(
                    [self.translation_probs[wt, ws] * self.alignment_probs[ self.get_jump(j, i-1, len_s, len_t) ]
                     for j, ws in enumerate(source)]
                )
            else:
                maxlink = np.argmax([self.translation_probs[wt, ws] for ws in source])
            if maxlink != 0:
                links.add((maxlink, i))
        return list(links)


def jump_size_in_errors(model, en_corpus_path: str, fr_corpus_path: str, alignment_path: str):

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
            # We only care about incorrect alignments
            if a in sure_a[idx]:
                continue
            if a in prob_a[idx]:
                continue

            # If the predicted alignment is incorrect...
            ae, af = a
            pred_distance = abs(af - ae)

            for (gold_ae, gold_af) in sure_a[idx]:
                if gold_af == af:
                    gold_distance = abs(gold_af - gold_ae)
                    # if gold_distance - pred_distance == 0:
                    #     print(a, (gold_ae, gold_af))
                    errors_sure[gold_distance - pred_distance] += 1
                    errors_all[gold_distance - pred_distance] += 1


            for (gold_ae, gold_af) in prob_a[idx]:
                if gold_af == af:
                    gold_distance = abs(gold_af - gold_ae)
                    # if gold_distance - pred_distance == 0:
                    #     print(a, (gold_ae, gold_af))
                    errors_prob[gold_distance - pred_distance] += 1
                    errors_all[gold_distance - pred_distance] += 1

    return errors_all, errors_sure, errors_prob


def offset_in_errors(model, abs_value, en_corpus_path: str, fr_corpus_path: str, alignment_path: str):

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
            # We only care about incorrect alignments
            if a in sure_a[idx]:
                continue
            if a in prob_a[idx]:
                continue

            # If the predicted alignment is incorrect...
            ae, af = a

            for (gold_ae, gold_af) in sure_a[idx]:
                if gold_af == af:
                    discrepancy = ae - gold_ae
                    if abs_value:
                        discrepancy = abs(discrepancy)
                    errors_sure[discrepancy] += 1
                    errors_all[discrepancy] += 1

            for (gold_ae, gold_af) in prob_a[idx]:
                if gold_af == af:
                    discrepancy = ae - gold_ae
                    if abs_value:
                        discrepancy = abs(discrepancy)
                    errors_prob[discrepancy] += 1
                    errors_all[discrepancy] += 1

    return errors_all, errors_sure, errors_prob


def plot_distance_errors(errors, ax=None):
    labels, values = zip(*sorted(errors.items(), key=lambda kv: kv[0]))
    indexes = np.arange(len(labels))
    width = 1

    if ax:
        ax.bar(labels, values, width)
    else:
        plt.bar(indexes, values, width)
        plt.xticks(indexes + width * 0.5, labels)
        plt.show()


if __name__ == '__main__':
    m1 = AlignmentModel('../pickles/translation_probs_IBM1.pickle')
    m2 = AlignmentModel('../pickles/translation_probs_IBM2.pickle', '../pickles/alignment_probs_IBM2.pickle')

    errors_all_1, _, _ = jump_size_in_errors(m1,
                                     '../data/test/test.e',
                                     '../data/test/test.f',
                                     '../data/test/test.wa.nonullalign')

    errors_all_2, _, _ = jump_size_in_errors(m2,
                                                 '../data/test/test.e',
                                                 '../data/test/test.f',
                                                 '../data/test/test.wa.nonullalign')

    # errors_all_1, _, _ = offset_in_errors(m1,
    #                                       False,
    #                                       '../data/test/test.e',
    #                                       '../data/test/test.f',
    #                                       '../data/test/test.wa.nonullalign')
    #
    # errors_all_2, _, _ = offset_in_errors(m2,
    #                                       False,
    #                                       '../data/test/test.e',
    #                                       '../data/test/test.f',
    #                                       '../data/test/test.wa.nonullalign')


    plt.rcParams["figure.figsize"] = [16, 9]
    plt.rcParams.update({'font.size': 26})

    f, (ax1, ax2) = plt.subplots(1, 2, sharex='all', sharey='all')

    # f.suptitle('Shared title')

    plot_distance_errors(errors_all_1, ax1)
    plot_distance_errors(errors_all_2, ax2)

    deltas_1 = list(errors_all_1.keys())
    deltas_2 = list(errors_all_2.keys())
    deltas = deltas_1 + deltas_2

    print("IBM1's error range: [{}, {}]".format(min(deltas_1), np.max(deltas_1)))
    print("IBM2's error range: [{}, {}]".format(min(deltas_2), np.max(deltas_2)))

    ax1.set_title('IBM 1')
    ax2.set_title('IBM 2')

    for ax in [ax1, ax2]:
        ax.label_outer()
        ax.xaxis.set_ticks(np.arange(np.min(deltas), np.max(deltas), 5))

    f.text(0.5, 0.01, 'Jump size', ha='center')
    f.text(0.05, 0.5, 'Errors', va='center', rotation='vertical')
    plt.savefig('jump-errors-test.pdf')
    # plt.show()