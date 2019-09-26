import pandas as pd

INPUT  = 'data/overextension_production_nouns.csv'
OUTPUT = 'intermediate/overextension_production_pairs_clean.csv'

df = pd.read_csv(INPUT)

# Keep relevant rows only.
df = df[
        (df['noun_to_noun']        == 1)      &
        (df['production_wordnet']  != 'NONE') &
        (df['production_imagenet'] != 'NONE') &
        (df['sense_wordnet']       != 'NONE') &
        (df['sense_imagenet']      != 'NONE') &
        (df['production_wordnet']  != df['sense_wordnet'])
        ]

# Populate synset -> word map (just for humans, i.e., information's sake).
df['production_word'] = None
df['sense_word']      = None
synset_to_word = {}
for i, row in df.iterrows():
    wordnet = row['production_wordnet']
    if wordnet not in synset_to_word:
        synset_to_word[wordnet] = row['norm_production']
    df.loc[i, 'production_word'] = synset_to_word[wordnet]

    wordnet = row['sense_wordnet']
    if wordnet not in synset_to_word:
        synset_to_word[wordnet] = row['norm_sense']
    df.loc[i, 'sense_word'] = synset_to_word[wordnet]

# Keep relevant columns only.
df = df[[
    'source_id',
    'production_word',
    'production_wordnet',
    'production_imagenet',
    'sense_word',
    'sense_wordnet',
    'sense_imagenet'
    ]]

# Deduplicate.
df['keep'] = True
seen_pairs = set()
for i, row in df.iterrows():
    pair = row['production_wordnet'], row['sense_wordnet']
    if pair in seen_pairs:
        df.loc[i, 'keep'] = False
    else:
        seen_pairs.add(pair)

df = df[df['keep']]
del df['keep']

df.to_csv(OUTPUT, index=False)
