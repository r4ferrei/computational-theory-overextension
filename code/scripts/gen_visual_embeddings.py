import os
import pickle

import pandas as pd

from imagenet import VisualEmbeddings
import common_flags

FULL_LEXICON = 'intermediate/full_lexicon.csv'
COMPREHENSION_SYNSETS = 'data/mcdonough_synsets.csv'

OUTPUT_EMBEDDINGS = 'intermediate/visual_embeddings.pkl'
OUTPUT_IMAGE_PATHS_DIR = 'intermediate/image-paths'

# Load existing file to reuse results.
# NOTE: delete file to re-generate.
try:
    with open(OUTPUT_EMBEDDINGS, 'rb') as f:
        dic = pickle.load(f)
except FileNotFoundError:
    dic = {}

embedder = VisualEmbeddings()

lexicon = pd.read_csv(FULL_LEXICON)

comprehension = pd.read_csv(COMPREHENSION_SYNSETS)
comprehension = comprehension[comprehension['imagenet_synset'] != 'NONE']

synsets = (
        set(lexicon['imagenet_synset'].values) |
        set(comprehension['imagenet_synset'].values))

for i, synset in enumerate(synsets):
    print("%d/%d" % (i+1, len(synsets)))

    if synset in dic:
        print("Already computed, skipping")
        continue

    embedding, paths = embedder.embedding_and_imagenet_paths_for_synset(
            synset)
    if embedding is not None:
        embedding = embedding.cpu().numpy()

    dic[synset] = embedding

    with open(os.path.join(OUTPUT_IMAGE_PATHS_DIR, synset), 'w') as f:
        for path in paths:
            print(path, file=f)

with open(OUTPUT_EMBEDDINGS, 'wb') as f:
    pickle.dump(dic, f)
