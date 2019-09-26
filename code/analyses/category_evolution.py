import pickle
import argparse

import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import model

parser = argparse.ArgumentParser()
parser.add_argument('--concepts', help="file with concept list")
args = parser.parse_args()

CONCEPTS = args.concepts
assert(CONCEPTS)

REL_FREQ      = 'intermediate/prod_vocab_rel_freqs.csv'
PROD_VOCAB    = 'intermediate/filtered_lexicon_mcdonough.csv' # attention!
FULL_LEXICON  = 'intermediate/filtered_lexicon_mcdonough.csv'
DIST_MATRIX   = 'intermediate/dist_matrix_square_mcdonough.npy'

# Load frequencies.
rel_freq_df = pd.read_csv(REL_FREQ)
rel_freq_dic = {}
for i, row in rel_freq_df.iterrows():
    rel_freq_dic[row['wordnet_synset']] = row['rel_freq']

# Compute reverse lookup tables for prod_vocab and full_lexicon.
# NOTE: keys are (wordnet, imagenet) synset name pairs.
prod_vocab = pd.read_csv(PROD_VOCAB)
prod_vocab_lookup = {}
for i, row in prod_vocab.iterrows():
    prod_vocab_lookup[row['wordnet_synset'], row['imagenet_synset']] = i

full_lexicon = pd.read_csv(FULL_LEXICON)
full_lexicon_lookup = {}
for i, row in full_lexicon.iterrows():
    full_lexicon_lookup[row['wordnet_synset'], row['imagenet_synset']] = i

rel_freqs = np.zeros((len(prod_vocab),))
for i, row in prod_vocab.iterrows():
    synset = row['wordnet_synset']
    rel_freqs[i] = rel_freq_dic[synset]

# Load concepts.
concepts = pd.read_csv(CONCEPTS)
lexicon_indices = []
for i, row in concepts.iterrows():
    lexicon_indices.append(
            full_lexicon_lookup[row['wordnet_synset'], row['imagenet_synset']])

# Prepare optimization.
dist_matrix = torch.tensor(np.load(DIST_MATRIX), device='cuda')
rel_freqs   = torch.tensor(rel_freqs, device='cuda')

# Loop over h parameters and predict category labels.

with open('intermediate/model_parameters.pkl', 'rb') as f:
    param = pickle.load(f)
    learned_h = float(param['h'])

MAX_H  = learned_h * 1.55
MIN_H  = 0.1
NUM_HS = 20

last_labels = []

def print_categories(h_val, concepts, labels):
    assert(len(concepts) == len(labels))

    label_concepts = {}
    for i in range(len(concepts)):
        label   = prod_vocab.loc[labels[i], 'wordnet_synset']
        concept = concepts.loc[i, 'wordnet_synset']

        if label not in label_concepts:
            label_concepts[label] = []
        label_concepts[label].append(concept)

    sorted_labels = sorted(list(label_concepts.keys()))

    print()
    print("h = %.3f" % h_val)
    for label in sorted_labels:
        print("%s:" % label)
        for concept in label_concepts[label]:
            print("\t%s" % concept)

data = []
for h_val in np.linspace(MAX_H, MIN_H, num=NUM_HS):
    # Parameters.
    h             = torch.tensor([h_val], dtype=rel_freqs.dtype, device='cuda')
    kernel_widths = model.compute_kernel_widths(h)
    priors        = model.rel_freqs_to_priors(rel_freqs, uniform=False)

    # Prediction.
    ranks, posts, full = model.predict_production_ranks_and_posteriors(
            prod_sense_dist       = dist_matrix,
            identity_prod_indices = [None] * len(full_lexicon),
            kernel_widths         = kernel_widths,
            priors                = priors,
            full_dist_matrix      = None,
            child_prod_indices    = None,
            allow_identity_prod   = True,
            target_identity_prod  = True,
            get_full_matrix       = True)

    labels = []
    for i in range(len(concepts)):
        labels.append(int(
            torch.argmax(full[:, lexicon_indices[i]]).detach().cpu()))

    if labels != last_labels:
        last_labels = labels
        print_categories(h_val, concepts, labels)
