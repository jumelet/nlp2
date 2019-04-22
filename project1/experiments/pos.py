import os
import pickle
import sys
from collections import Counter, defaultdict
from pprint import pprint

import numpy as np


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
    align = defaultdict(list)
    align_s = defaultdict(list)
    align_p = defaultdict(list)
    with open(alignment_path) as f:
        lines = [l.strip().split(' ') for l in f.read().strip().split('\n')]
        for l in lines:
            align[int(l[0])].append((int(l[1]), int(l[2])))
            if l[3] == 'S':
                align_s[int(l[0])].append((int(l[1]), int(l[2])))
            else:
                align_p[int(l[0])].append((int(l[1]), int(l[2])))

    return align, align_s, align_p


def calc_pos_alignments(en, fr, align_s, align_p, enpos, frpos, probs_path: str):
    """
    Calculates the alignments and maps these to the corresponding POS tags.
    Returns 3 dictionaries, corresponding to the sure, possible and wrong alignments.
    """
    sys.path.append(os.path.expanduser('..'))
    with open(probs_path, 'rb') as file:
        probs = pickle.load(file)

    def get_alignments(s, t):
        a = []
        for i, wt in enumerate(t, start=1):
            maxlink = np.argmax([probs[wt, ws] for ws in s])
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
                pos_wrong[a_pos] += 1

    return pos_s, pos_p, pos_wrong


def calc_pos_sequence_alignments(seq_len, en, fr, align_s, align_p, enpos, frpos, probs_path: str):
    """
    Calculates the alignments and maps these to the corresponding sequences of POS tags.
    Returns 3 dictionaries, corresponding to the sure, possible and wrong alignments.
    (The keys of the dictionaries are POS tag sequences such as 'DET-NOUN-VERB'.)
    """
    sys.path.append(os.path.expanduser('..'))
    with open(probs_path, 'rb') as file:
        probs = pickle.load(file)

    def get_alignments(s, t):
        a = []
        for i, wt in enumerate(t, start=1):
            maxlink = np.argmax([probs[wt, ws] for ws in s])
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


if __name__ == '__main__':
    en_pos, en_sen = read_pos('../data/tagged/dev.pos.e', '../data/validation/dev.e')
    fr_pos, fr_sen = read_pos('../data/tagged/dev.pos.f', '../data/validation/dev.f')

    all_a, sure_a, prob_a = read_alignments('../data/validation/dev.wa.nonullalign')

    sure_pos_a, prob_pos_a, wrong_pos_a = calc_pos_alignments(en_sen, fr_sen,
                                                              sure_a, prob_a,
                                                              en_pos, fr_pos,
                                                              '../pickles/translation_probs_IBM1.pickle')

    mc = 10
    print('10 most common pos2pos in sure links:')
    pprint(sure_pos_a.most_common(mc))

    print('\n10 most common pos2pos in probable-sure links:')
    pprint(prob_pos_a.most_common(mc))

    print('\n10 most common wrong pos2pos links')
    pprint(wrong_pos_a.most_common(mc))

    en_seq_s, en_seq_p, en_seq_wrong, fr_seq_s, fr_seq_p, fr_seq_wrong = calc_pos_sequence_alignments(
                                                                            3,
                                                                            en_sen, fr_sen,
                                                                            sure_a, prob_a,
                                                                            en_pos, fr_pos,
                                                                            '../pickles/translation_probs_IBM1.pickle')

    print('The 10 most common POS sequences in sure links:')
    pprint(en_seq_s.most_common(mc))
    # pprint(fr_seq_s.most_common(mc))

    print('\nThe 10 most common POS sequences in probable-sure links:')
    pprint(en_seq_p.most_common(mc))
    # pprint(fr_seq_p.most_common(mc))

    print('\nThe 10 POS sequences containing most wrong links')
    pprint(en_seq_wrong.most_common(mc))
    # pprint(fr_seq_wrong.most_common(mc))
