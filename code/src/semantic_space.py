import pickle
import numpy as np
import pandas as pd

import pandas as pd
from nltk.corpus import wordnet as wn

import swow as swow_module
import common_flags

VISUAL_EMBEDDINGS = 'intermediate/visual_embeddings.pkl'

precomputed_dists = None
precomputed_wordnet = None
precomputed_imagenet = None

def load_precomputed(dists, vocab):
    global precomputed_dists
    global precomputed_wordnet
    global precomputed_imagenet

    precomputed_dists = np.load(dists)

    precomputed_wordnet = {}
    precomputed_imagenet = {}

    vocab = pd.read_csv(vocab)
    for i, row in vocab.iterrows():
        precomputed_wordnet[row['wordnet_synset']] = i
        precomputed_imagenet[row['imagenet_synset']] = i

def get_precomputed(u, v, precomputed_index, dim):
    return precomputed_dists[
            precomputed_index[u],
            precomputed_index[v],
            dim]

try:
    swow = swow_module.SWOW()
except:
    swow = None # if only using precomputed in analyses, this is OK

with open(VISUAL_EMBEDDINGS, 'rb') as f:
    visual_embeddings = pickle.load(f)

def _cos_dist(u, v):
    return 1 - np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)

# All functions take names of synsets.

# Clamp values to prevent numerical issues.

def categorical_distance(u, v):
    if precomputed_dists is None:
        return max(0, 1 - wn.wup_similarity(wn.synset(u), wn.synset(v)))
    else:
        return get_precomputed(u, v, precomputed_wordnet, 0)

def visual_distance(u, v):
    if precomputed_dists is None:
        return max(0, _cos_dist(visual_embeddings[u], visual_embeddings[v]))
    else:
        return get_precomputed(u, v, precomputed_imagenet, 1)

def predicate_distance(u, v):
    if precomputed_dists is None:
        return max(0, swow.distance(u, v))
    else:
        return get_precomputed(u, v, precomputed_wordnet, 2)
