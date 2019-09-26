import pandas as pd
from nltk.corpus import wordnet as wn

import common_flags

OVEREXTENSION = 'intermediate/overextension_production_pairs_clean.csv'
WORDBANK      = 'data/wordbank_vocab.csv'

OUTPUT_VOCAB   = 'intermediate/productive_vocab.csv'
OUTPUT_LEXICON = 'intermediate/full_lexicon.csv'

if common_flags.is_mcdonough():
    OUTPUT_VOCAB    = 'intermediate/productive_vocab_mcdonough.csv'
    OUTPUT_LEXICON  = 'intermediate/full_lexicon_mcdonough.csv'
    BOOKLET         = 'intermediate/filtered_booklets.csv'
    BOOKLET_SYNSETS = 'data/mcdonough_synsets.csv'

    booklet = pd.read_csv(BOOKLET)
    booklet_items = list(
            set(booklet['early_high'].values) |
            set(booklet['early_low'].values)  |
            set(booklet['late_low'].values))
    booklet_synsets = pd.read_csv(BOOKLET_SYNSETS)

    MCDONOUGH_REMOVE = [] # not used anymore

overextension_df = pd.read_csv(OVEREXTENSION)
wordbank_df      = pd.read_csv(WORDBANK)

# Keep only Wordbank vocabulary with features.
wordbank_df = wordbank_df[
        (wordbank_df['wordnet_synset']  != 'NONE') &
        (wordbank_df['imagenet_synset'] != 'NONE')
        ]

def zip_cols(df, col1, col2):
    return [(row[col1], row[col2]) for i, row in df.iterrows()]

def standardize_synsets(pairs):
    def doit(name):
        return wn.synset(name).name()
    return [(doit(a), doit(b)) for a, b in pairs]

# Capture all synsets with their standard WordNet names.
overextension_production_items = standardize_synsets(zip_cols(
        overextension_df, 'production_wordnet', 'production_imagenet'))
overextension_sense_items = standardize_synsets(zip_cols(
        overextension_df, 'sense_wordnet', 'sense_imagenet'))
wordbank_items = standardize_synsets(zip_cols(
        wordbank_df, 'wordnet_synset', 'imagenet_synset'))

# Assemble (wordnet_synset, imagenet_synset) pairs for McDonough booklet data.
if common_flags.is_mcdonough():
    booklet_dic = {}
    for i, row in booklet_synsets.iterrows():
        booklet_dic[row['wordnet_synset']] = row['imagenet_synset']
    booklet_items = [
            (wordnet, booklet_dic[wordnet])
            for wordnet in booklet_items
            ]
    booklet_items = standardize_synsets(booklet_items)

# Assert data integrity (unique WordNet -> ImageNet mapping).
wordnet_to_imagenet = {}
for pair in (
        overextension_production_items +
        overextension_sense_items +
        wordbank_items):
    prod, sense = pair
    if prod in wordnet_to_imagenet:
        assert(sense == wordnet_to_imagenet[prod])
    else:
        wordnet_to_imagenet[prod] = sense

def df_from_pairs(pairs):
    return pd.DataFrame({
        'wordnet_synset'  : [a for a, b in pairs],
        'imagenet_synset' : [b for a, b in pairs]
        })

# Productive vocabulary = Wordbank U all overextension productions.
if common_flags.is_mcdonough():
    prod_vocab = (
            set(overextension_production_items) |
            set(wordbank_items)                 |
            set(booklet_items))

    # Remove confounders from McDonough productive vocabulary.
    prod_vocab = [(a, b) for a, b in prod_vocab if a not in MCDONOUGH_REMOVE]
else:
    prod_vocab = (
            set(overextension_production_items) |
            set(wordbank_items))

prod_vocab = sorted(list(prod_vocab))
df_from_pairs(prod_vocab).to_csv(OUTPUT_VOCAB, index=False)

# Full lexicon = Wordank U overextension productions U overextension senses.
if common_flags.is_mcdonough():
    full_lexicon = (
            set(overextension_production_items) |
            set(overextension_sense_items)      |
            set(wordbank_items)                 |
            set(booklet_items))
else:
    full_lexicon = (
            set(overextension_production_items) |
            set(overextension_sense_items)      |
            set(wordbank_items))

full_lexicon = sorted(list(full_lexicon))
df_from_pairs(full_lexicon).to_csv(OUTPUT_LEXICON, index=False)
