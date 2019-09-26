import os

import pandas as pd
import pylangacq as pla
from nltk.stem import WordNetLemmatizer

CHILDES = 'data/childes/'
OUTPUT_COUNTS = 'intermediate/childes_noun_counts.csv'
OUTPUT_PATHS  = 'intermediate/childes_transcripts_used.csv'

# From Clark (1973).
MIN_AGE = 13
MAX_AGE = 30

with open('data/all_corpora_paths.txt', 'r') as f:
    filenames = [line.strip() for line in f]
    filepaths = [os.path.join(CHILDES, fname) for fname in filenames]

# Filter by age and no parsing errors.
good_filepaths = []
for i, fpath in enumerate(filepaths):
    try:
        print("Pre-reading %d/%d" % (i+1, len(filepaths)))
        reader = pla.read_chat(fpath)

        age = reader.age(months=True)
        age = list(age.values())[0]

        if MIN_AGE <= age <= MAX_AGE:
            good_filepaths.append(fpath)

    except Exception as e:
        print("Warning: failed to read %s" % fpath)
        print(e)
        print()

# Count nouns.
noun_counts = {}

print("Parsing %d files" % len(good_filepaths))
reader = pla.read_chat(*good_filepaths)
tagged_words = reader.tagged_words(exclude='CHI')

print("Counting frequencies")
lemmatizer = WordNetLemmatizer()
for word, pos, mor, rel in tagged_words:
    if pos not in ['N', 'N:PT']: continue
    lemma = lemmatizer.lemmatize(word)
    noun_counts[lemma] = noun_counts.get(lemma, 0) + 1

df = pd.DataFrame({
    'word'  : list(noun_counts.keys()),
    'count' : list(noun_counts.values())
    })
df.to_csv(OUTPUT_COUNTS, index=False)

fpaths_df = pd.DataFrame({'filepath': good_filepaths})
fpaths_df.to_csv(OUTPUT_PATHS, index=False)
