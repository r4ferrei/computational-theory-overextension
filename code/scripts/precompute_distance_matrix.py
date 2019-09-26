import pickle
import numpy as np

import pandas as pd
from nltk.corpus import wordnet as wn

import semantic_space
import common_flags

# NOTE: dimensions are wordnet, imagenet, swow, i.e.,
# categorical, analogical, predicate-based.

PROD_VOCAB        = 'intermediate/filtered_prod_vocab.csv'
FULL_LEXICON      = 'intermediate/filtered_lexicon.csv'

OUTPUT            = 'intermediate/dist_matrix.npy'

if common_flags.is_mcdonough():
    PROD_VOCAB        = 'intermediate/filtered_prod_vocab_mcdonough.csv'
    FULL_LEXICON      = 'intermediate/filtered_lexicon_mcdonough.csv'
    OUTPUT            = 'intermediate/dist_matrix_mcdonough.npy'
elif common_flags.is_square():
    PROD_VOCAB   = FULL_LEXICON
    OUTPUT       = 'intermediate/dist_matrix_square.npy'
elif common_flags.is_square_mcdonough():
    FULL_LEXICON = 'intermediate/filtered_lexicon_mcdonough.csv'
    PROD_VOCAB   = FULL_LEXICON
    OUTPUT       = 'intermediate/dist_matrix_square_mcdonough.npy'

prod_vocab   = pd.read_csv(PROD_VOCAB)
full_lexicon = pd.read_csv(FULL_LEXICON)

print("Categorical distances")
cat_dist = np.zeros((len(prod_vocab), len(full_lexicon)))
for i in range(len(prod_vocab)):
    print("%d/%d" % (i+1, len(prod_vocab)))

    for j in range(len(full_lexicon)):
        u = prod_vocab.loc[i, 'wordnet_synset']
        v = full_lexicon.loc[j, 'wordnet_synset']

        cat_dist[i, j] = semantic_space.categorical_distance(u, v)

def cos_dist(u, v):
    return 1 - np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)

print("Visual-analogical distances")
vis_dist = np.zeros((len(prod_vocab), len(full_lexicon)))
for i in range(len(prod_vocab)):
    print("%d/%d" % (i+1, len(prod_vocab)))

    for j in range(len(full_lexicon)):
        u = prod_vocab.loc[i, 'imagenet_synset']
        v = full_lexicon.loc[j, 'imagenet_synset']

        vis_dist[i, j] = semantic_space.visual_distance(u, v)

print("Predicate-based distances")
pred_dist = np.zeros((len(prod_vocab), len(full_lexicon)))
for i in range(len(prod_vocab)):
    print("%d/%d" % (i+1, len(prod_vocab)))

    for j in range(len(full_lexicon)):
        u = prod_vocab.loc[i, 'wordnet_synset']
        v = full_lexicon.loc[j, 'wordnet_synset']

        pred_dist[i, j] = semantic_space.predicate_distance(u, v)

dist_matrix = np.stack((cat_dist, vis_dist, pred_dist), axis=2)

np.save(OUTPUT, dist_matrix)
