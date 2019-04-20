import os
import pickle
import sys
from collections import Counter, defaultdict
from pprint import pprint

import numpy as np


def read_pos(pos_path: str, corpus_path: str):
    with open(pos_path, 'r') as f:
        pos = [p.strip().split(' ') for p in f.read().strip().split('\n')]

    with open(corpus_path, 'r') as f:
        corpus = [s.strip().split(' ') for s in f.read().strip().split('\n')]

    pos = {i: c for i, c in enumerate(pos, start=1)}
    corpus = {i: c for i, c in enumerate(corpus, start=1)}

    return pos, corpus


def read_alignments(alignment_path: str):
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


def calc_pos_alignments(en, fr, align_s, align_p, enpos, frpos):
    sys.path.append(os.path.expanduser('..'))
    with open('../pickles/translation_probs_IBM1.pickle', 'rb') as f:
        probs = pickle.load(f)

    def get_alignments(s, t):
        a = []
        for i, wt in enumerate(t, start=1):
            a.append((np.argmax([probs[wt, ws] for ws in s]), i))
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


en_pos, en_sen = read_pos('../data/tagged/dev.pos.e', '../data/validation/dev.e')
fr_pos, fr_sen = read_pos('../data/tagged/dev.pos.f', '../data/validation/dev.f')

all_a, sure_a, prob_a = read_alignments('../data/validation/dev.wa.nonullalign')

sure_pos_a, prob_pos_a, wrong_pos_a = calc_pos_alignments(en_sen, fr_sen, sure_a, prob_a, en_pos, fr_pos)

mc = 10
print('10 most common pos2pos in sure links:')
pprint(sure_pos_a.most_common(mc))

print('\n10 most common pos2pos in probable-sure links:')
pprint(prob_pos_a.most_common(mc))

print('\n10 most common wrong pos2pos links')
pprint(wrong_pos_a.most_common(mc))
