import spacy
from tqdm import tqdm

from project1.data_reader import DataReader


###
source_path = 'data/training/hansards.36.2.e'
target_path = 'data/training/hansards.36.2.f'

source_output_path = 'data/training/hansards.36.2.pos.e'
target_output_path = 'data/training/hansards.36.2.pos.f'
###

En = spacy.load('en_core_web_md')
Fr = spacy.load('fr_core_news_md')

data_reader = DataReader(source_path, target_path)

with open(source_output_path, 'w') as source_f_out, open(target_output_path, 'w') as target_f_out:
    for (e, f) in tqdm(data_reader.get_parallel_data(), total=len(data_reader)):

        en_doc = En(' '.join(e))
        fr_doc = Fr(' '.join(f))

        en_pos_tags = [token.pos_ for token in en_doc]
        fr_pos_tags = [token.pos_ for token in fr_doc]

        print(' '.join(en_pos_tags), file=source_f_out)
        print(' '.join(fr_pos_tags), file=target_f_out)