from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import pandas as pd

import common_flags

# NOTE: relative to all nouns in CDS, not just within productive vocabulary
# being analyzed.

CHILDES_COUNTS = 'intermediate/childes_noun_counts.csv'
PROD_VOCAB     = 'intermediate/productive_vocab.csv'
FULL_LEXICON   = 'intermediate/full_lexicon.csv'
BOOKLET        = 'data/mcdonough_synsets.csv'

OUTPUT = 'intermediate/prod_vocab_rel_freqs.csv'

IGNORE = [('cable_car.n.01', 'car')]

# Load vocabulary.
prod_vocab   = pd.read_csv(PROD_VOCAB)
full_lexicon = pd.read_csv(FULL_LEXICON)
booklet      = pd.read_csv(BOOKLET)

# Collect CHILDES counts.
childes = pd.read_csv(CHILDES_COUNTS)
rel_counts  = {}
total_count = 0
for i, row in childes.iterrows():
    word = row['word']
    count = row['count']
    rel_counts[word] = count
    total_count += count

total_count += len(rel_counts) # add-one smoothing

for word in rel_counts:
    rel_counts[word] += 1 # add-one smoothing
    rel_counts[word] /= total_count

# Compute total relative frequency per synset, across lemmas.
lemmatizer = WordNetLemmatizer()

synsets_to_measure = list(prod_vocab['wordnet_synset'].values)
elements = set(synsets_to_measure)
for synset_name in booklet['wordnet_synset'].values:
    if synset_name not in elements:
        synsets_to_measure.append(synset_name)
        elements.add(synset_name)
for synset_name in full_lexicon['wordnet_synset'].values:
    if synset_name not in elements:
        synsets_to_measure.append(synset_name)
        elements.add(synset_name)

rel_freq = {}
for synset_name in synsets_to_measure:
    total = 0
    for name in wn.synset(synset_name).lemma_names():
        if (synset_name, name) in IGNORE:
            continue

        lemma = lemmatizer.lemmatize(name)
        lemma = lemma.replace('_', '') # PyLangAcq treatment of compund nouns
        total += rel_counts.get(lemma, 0)
    rel_freq[synset_name] = max(1/total_count, total) # add-one smoothing

# Save result.
df = pd.DataFrame({
    'wordnet_synset' : list(rel_freq.keys()),
    'rel_freq'       : list(rel_freq.values()),
    })
df.to_csv(OUTPUT, index=False)
