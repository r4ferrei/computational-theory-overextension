import pickle

import pandas as pd

import common_flags
import swow as swow_module

INPUT_OVEREXTENSION = 'intermediate/overextension_production_pairs_clean.csv'
OUTPUT_OVEREXTENSION = 'intermediate/filtered_overextension_pairs.csv'

INPUT_PROD_VOCAB = 'intermediate/productive_vocab.csv'
OUTPUT_PROD_VOCAB = 'intermediate/filtered_prod_vocab.csv'

INPUT_FULL_LEXICON = 'intermediate/full_lexicon.csv'
OUTPUT_FULL_LEXICON = 'intermediate/filtered_lexicon.csv'

VISUAL_EMBEDDINGS = 'intermediate/visual_embeddings.pkl'

if common_flags.is_mcdonough():
    INPUT_PROD_VOCAB = 'intermediate/productive_vocab_mcdonough.csv'
    OUTPUT_PROD_VOCAB = 'intermediate/filtered_prod_vocab_mcdonough.csv'

    INPUT_FULL_LEXICON = 'intermediate/full_lexicon_mcdonough.csv'
    OUTPUT_FULL_LEXICON = 'intermediate/filtered_lexicon_mcdonough.csv'

overextension = pd.read_csv(INPUT_OVEREXTENSION)
prod_vocab    = pd.read_csv(INPUT_PROD_VOCAB)
full_lexicon  = pd.read_csv(INPUT_FULL_LEXICON)

swow = swow_module.SWOW()

with open(VISUAL_EMBEDDINGS, 'rb') as f:
    visual_embeddings = pickle.load(f)

def filter_df_on_cols(df, wordnet_cols, imagenet_cols):
    df['has_features'] = True
    for i in range(len(df)):
        for col in wordnet_cols:
            synset = df.loc[i, col]
            if swow.distance(synset, synset) is None:
                df.loc[i, 'has_features'] = False
        for col in imagenet_cols:
            synset = df.loc[i, col]
            if visual_embeddings.get(synset) is None:
                df.loc[i, 'has_features'] = False
    df = df[df['has_features']]
    del df['has_features']
    return df

overextension = filter_df_on_cols(overextension,
        wordnet_cols  = ['production_wordnet',  'sense_wordnet'],
        imagenet_cols = ['production_imagenet', 'sense_imagenet'])

prod_vocab = filter_df_on_cols(prod_vocab,
        wordnet_cols  = ['wordnet_synset'],
        imagenet_cols = ['imagenet_synset'])

full_lexicon = filter_df_on_cols(full_lexicon,
        wordnet_cols  = ['wordnet_synset'],
        imagenet_cols = ['imagenet_synset'])

overextension.to_csv(OUTPUT_OVEREXTENSION, index=False)
prod_vocab.to_csv(OUTPUT_PROD_VOCAB, index=False)
full_lexicon.to_csv(OUTPUT_FULL_LEXICON, index=False)
