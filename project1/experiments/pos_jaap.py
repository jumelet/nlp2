import os
import pickle
import sys
from collections import Counter, defaultdict
from pprint import pprint
from project1.aer import read_naacl_alignments, AERSufficientStatistics

import numpy as np


sys.path.append(os.path.expanduser('..'))


# POS & ALIGNMENT READING #
def read_pos(pos_path: str, corpus_path: str):
    """ Reads the pos tags and original corpus of source & target. """
    with open(pos_path, 'r') as f:
        pos = [p.strip().split(' ') for p in f.read().strip().split('\n')]

    with open(corpus_path, 'r') as f:
        corpus = [s.strip().split(' ') for s in f.read().strip().split('\n')]

    pos = {i: c for i, c in enumerate(pos, start=1)}
    corpus = {i: c for i, c in enumerate(corpus, start=1)}

    return pos, corpus


def read_alignments(alignment_path: str):
    """ Reads the gold alignments. Returns 3 dictionary, containing all,
    sure and possible alignemts. Dictionaries map a sentence key to a list of
    (a_s, a_t) links.
    """
    align = defaultdict(set)
    align_s = defaultdict(set)
    align_p = defaultdict(set)
    with open(alignment_path) as f:
        lines = [l.strip().split(' ') for l in f.read().strip().split('\n')]
        for l in lines:
            align[int(l[0])].add((int(l[1]), int(l[2])))
            if l[3] == 'S':
                align_s[int(l[0])].add((int(l[1]), int(l[2])))
                align_p[int(l[0])].add((int(l[1]), int(l[2])))  # S also in P!
            else:
                align_p[int(l[0])].add((int(l[1]), int(l[2])))

    return align, align_s, align_p


def get_jump(e_pos: int, f_pos: int, len_e: int, len_f: int) -> int:
    return e_pos - np.floor(f_pos * (len_e / len_f))


def calc_ef_prob(wf: str, we: str, e_pos: int, f_pos: int, len_e: int, len_f: int):
    if IBM_TYPE == 'IBM2':
        jump = get_jump(e_pos, f_pos, len_e, len_f)
        return PROBS[wf, we] * ALIGN_PROBS[jump]

    return PROBS[wf, we]


# POS ALIGNMENTS #
def calc_pos_alignments(en, fr, align_s, align_p, enpos, frpos):
    """
    Calculates the alignments and maps these to the corresponding POS tags.
    Returns 3 dictionaries, corresponding to the sure, possible and wrong alignments.
    """

    def get_alignments(s, t):
        a = []
        len_s = len(s)
        len_t = len(t)
        for i, wt in enumerate(t, start=1):
            maxlink = np.argmax(
                [calc_ef_prob(wt, ws, j, i, len_s, len_t) for j, ws in enumerate(s)]
            )
            if maxlink != 0:
                a.append((maxlink, i))
        return a

    pos_s = Counter()
    pos_p = Counter()
    pos_wrong = Counter()

    for idx in en.keys():
        e = ['NULL'] + en[idx]
        f = fr[idx]
        alignments = get_alignments(e, f)

        a_s = align_s[idx]
        a_p = align_p[idx]

        for a in alignments:
            a_e, a_f = a
            a_pos = enpos[idx][a_e-1], frpos[idx][a_f-1]
            if a in a_s:
                pos_s[a_pos] += 1
            elif a in a_p:
                pos_p[a_pos] += 1
            else:
                if a_pos == ('ADP', 'DET'):
                    print(e[a_e], f[a_f - 1])
                pos_wrong[a_pos] += 1

    return pos_s, pos_p, pos_wrong


def calc_multiple_accuracy(en, fr, align_p):
    vocab_e = set()
    vocab_f = set()
    for wf, we in PROBS.keys():
        vocab_e.add(we)
        vocab_f.add(wf)

    n_mult = 0
    n_total = 0
    cor_mult = 0

    defaultprob = PROBS['bla', 'bla']

    # The boy    and the dog
    #  |
    #  |--------------
    #  |             |
    #  v             v
    # Le  garçon et  le  chien

    def get_alignments(s, t):
        a = set()
        mult_a = set()
        maxlinks = []
        len_s = len(s)
        len_t = len(t)

        for i, wt in enumerate(t, start=1):
            maxlink = np.argmax(
                [
                    (calc_ef_prob(wt, ws, j, i, len_s, len_t))  # - int(j in maxlinks and (ws in s[:j] or ws in s[j+1:]))*1e-12)
                    for j, ws in enumerate(s)
                ]
            )

            if maxlink != 0 and ((s[maxlink] in s[maxlink+1:] or s[maxlink] in s[:maxlink]) and maxlink in maxlinks):
                mult_a.add((maxlink, i))
            elif maxlink != 0:
                a.add((maxlink, i))
            maxlinks.append(maxlink)
        return a, mult_a

    for idx in en.keys():
        e = ['NULL'] + en[idx]
        f = fr[idx]
        alignments, mult_alignments = get_alignments(e, f)

        n_total += len(alignments) + len(mult_alignments)
        n_mult += len(mult_alignments)
        cor_mult += len(mult_alignments & set(align_p[idx]))

    print(cor_mult, n_mult, n_total, cor_mult / n_mult, n_mult/n_total)


def calc_pos_sequence_alignments(seq_len, en, fr, align_s, align_p, enpos, frpos):
    """
    Calculates the alignments and maps these to the corresponding sequences of POS tags.
    Returns 3 dictionaries, corresponding to the sure, possible and wrong alignments.
    (The keys of the dictionaries are POS tag sequences such as 'DET-NOUN-VERB'.)
    """
    def get_alignments(s, t):
        a = []
        for i, wt in enumerate(t, start=1):
            maxlink = np.argmax([PROBS[wt, ws] for ws in s])
            if maxlink != 0:
                a.append((maxlink, i))
        return a

    en_seq_s = Counter()
    en_seq_p = Counter()
    en_seq_wrong = Counter()

    fr_seq_s = Counter()
    fr_seq_p = Counter()
    fr_seq_wrong = Counter()

    seq_range = range(-(seq_len // 2 + 1), seq_len // 2)
    for idx in en.keys():
        e = ['NULL'] + en[idx]
        f = fr[idx]
        alignments = get_alignments(e, f)

        a_s = align_s[idx]
        a_p = align_p[idx]

        for a in alignments:
            a_e, a_f = a

            try:
                en_seq = '-'.join([enpos[idx][a_e+offset] for offset in seq_range])
                if a in a_s:
                    en_seq_s[en_seq] += 1
                elif a in a_p:
                    en_seq_p[en_seq] += 1
                else:
                    en_seq_wrong[en_seq] += 1
                    # print(en_seq)
                    # print([e[a_e+offset+1] for offset in seq_range])
                    # print([f[a_f+offset] for offset in seq_range])
                    # print(e)
                    # print(f)
            except IndexError:
                pass

            try:
                fr_seq = '-'.join([frpos[idx][a_f+offset] for offset in seq_range])
                if a in a_s:
                    fr_seq_s[fr_seq] += 1
                elif a in a_p:
                    fr_seq_p[fr_seq] += 1
                else:
                    fr_seq_wrong[fr_seq] += 1
            except IndexError:
                pass

    return en_seq_s, en_seq_p, en_seq_wrong, fr_seq_s, fr_seq_p, fr_seq_wrong


def create_test_file():
    def get_alignments(s, t):
        a = set()
        maxlinks = []
        len_s = len(s)
        len_t = len(t)

        for i, wt in enumerate(t, start=1):
            maxlink = np.argmax(
                [
                    (calc_ef_prob(wt, ws, j, i, len_s, len_t)) # - int(j in maxlinks and (ws in s[:j] or ws in s[j+1:]))*1e-12)
                    for j, ws in enumerate(s)
                ]
            )

            if maxlink != 0:
                a.add((maxlink, i))
            maxlinks.append(maxlink)
        return a

    en_pos, en_sen = read_pos('../data/tagged/test.pos.e', '../data/testing/test/test.e')
    fr_pos, fr_sen = read_pos('../data/tagged/test.pos.f', '../data/testing/test/test.f')

    metric = AERSufficientStatistics()

    gold_links = read_naacl_alignments('../data/testing/answers/test.wa.nonullalign')

    with open('../data/testing/eval/test_align.txt', 'w+') as f:
        for i, (s, t) in zip(en_sen.keys(), zip(en_sen.values(), fr_sen.values())):
            aligns = get_alignments(['NULL']+s, t)
            metric.update(sure=sure_a[i], probable=prob_a[i], predicted=aligns)
            for align in aligns:
                f.write(f'{"0"*(4-len(str(i)))}{i} {align[0]} {align[1]}\n')

    print(metric.aer())


if __name__ == '__main__':
    # MODEL PROBS READING #
    IBM_TYPE = 'IBM2'

    with open(f'../pickles/translation_probs_{IBM_TYPE}.pickle', 'rb') as file:
        PROBS = pickle.load(file)
    if IBM_TYPE == 'IBM2':
        with open('../pickles/alignment_probs_IBM2.pickle', 'rb') as file:
            ALIGN_PROBS = pickle.load(file)

    experiments = {
        'pos_alignments': False,
        'pos_sequence_alignments': False,
        'multiple_tokens':  False,
    }
    en_pos, en_sen = read_pos('../data/tagged/test.pos.e', '../data/testing/test/test.e')
    fr_pos, fr_sen = read_pos('../data/tagged/test.pos.f', '../data/testing/test/test.f')

    all_a, sure_a, prob_a = read_alignments('../data/testing/answers/test.wa.nonullalign')

    create_test_file()

    if experiments['pos_alignments']:
        sure_pos_a, prob_pos_a, wrong_pos_a = calc_pos_alignments(en_sen, fr_sen,
                                                                  sure_a, prob_a,
                                                                  en_pos, fr_pos)

        IBM_TYPE = 'IBM1'
        del PROBS
        with open(f'../pickles/translation_probs_{IBM_TYPE}.pickle', 'rb') as file:
            PROBS = pickle.load(file)

        sure_pos_a1, prob_pos_a1, wrong_pos_a1 = calc_pos_alignments(en_sen, fr_sen,
                                                                     sure_a, prob_a,
                                                                     en_pos, fr_pos)

        sure_pos_diff = Counter()
        prob_pos_diff = Counter()
        wrong_pos_diff = Counter()

        for k, v in sure_pos_a.items():
            sure_pos_diff[k] = v - sure_pos_a1.get(k, 0)
        for k in sure_pos_a1.keys() - sure_pos_a.keys():
            sure_pos_diff[k] = -sure_pos_a1[k]

        for k, v in prob_pos_a.items():
            prob_pos_diff[k] = v - prob_pos_a1.get(k, 0)
        for k in prob_pos_a1.keys() - prob_pos_a.keys():
            prob_pos_diff[k] = -prob_pos_a1[k]

        for k, v in wrong_pos_a.items():
            wrong_pos_diff[k] = v - wrong_pos_a1.get(k, 0)
        for k in wrong_pos_a1.keys() - wrong_pos_a.keys():
            wrong_pos_diff[k] = -wrong_pos_a1[k]

        mc = None
        print('most common pos2pos in sure links:')
        pprint([(k, v, sure_pos_a.get(k, 0), sure_pos_a1.get(k, 0))
                for k, v in sure_pos_diff.most_common(mc)])

        print('\nmost common pos2pos in probable-sure links:')
        pprint([(k, v, prob_pos_a.get(k, 0), prob_pos_a1.get(k, 0))
                for k, v in prob_pos_diff.most_common(mc)])

        print('\nmost common wrong pos2pos links')
        pprint([(k, v, wrong_pos_a.get(k, 0), wrong_pos_a1.get(k, 0))
                for k, v in wrong_pos_diff.most_common(mc)])

    if experiments['pos_sequence_alignments']:
        en_seq_s, en_seq_p, en_seq_wrong, fr_seq_s, fr_seq_p, fr_seq_wrong = calc_pos_sequence_alignments(
                                                                                3,
                                                                                en_sen, fr_sen,
                                                                                sure_a, prob_a,
                                                                                en_pos, fr_pos)

        mc = 10
        print('The 10 most common POS sequences in sure links:')
        pprint(en_seq_s.most_common(mc))
        # pprint(fr_seq_s.most_common(mc))

        print('\nThe 10 most common POS sequences in probable-sure links:')
        pprint(en_seq_p.most_common(mc))
        # pprint(fr_seq_p.most_common(mc))

        print('\nThe 10 POS sequences containing most wrong links')
        pprint(en_seq_wrong.most_common(mc))
        # pprint(fr_seq_wrong.most_common(mc))

    if experiments['multiple_tokens']:
        calc_multiple_accuracy(en_sen, fr_sen, prob_a)
