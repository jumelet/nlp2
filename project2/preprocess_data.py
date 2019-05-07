# PATHS
# training : 'project_2_data/02-21.10way.clean'
# validation: 'project_2_data/22.auto.clean'
# test: 'project_2_data/23.auto.clean'


def preprocess_to_sent(line):

    sent = []
    for i, sym in enumerate(line):
        if sym == ')' and line[i - 1] != ')' and line[i - 1] != ' ':
            word_start = 0
            c = None
            while c != ' ':

                c = line[i - word_start]
                word_start += 1

            word = line[i - word_start + 2:i]

            sent.append(word)
    return sent


def preprocess_data(path):
    corpus = []

    with open(path) as f:
        for line in f:
            sent = preprocess_to_sent(line)
            corpus.append(sent)

    return corpus
