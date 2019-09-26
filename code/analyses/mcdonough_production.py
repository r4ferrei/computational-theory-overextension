import pickle
import argparse

import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import model

parser = argparse.ArgumentParser()
parser.add_argument('--uniform', action='store_true', dest='uniform')
args = parser.parse_args()

UNIFORM = args.uniform

REL_FREQ        = 'intermediate/prod_vocab_rel_freqs.csv'
PROD_VOCAB      = 'intermediate/filtered_prod_vocab_mcdonough.csv'
FULL_LEXICON    = 'intermediate/filtered_lexicon_mcdonough.csv'
DIST_MATRIX     = 'intermediate/dist_matrix_mcdonough.npy'
BOOKLET         = 'intermediate/filtered_booklets.csv'
BOOKLET_SYNSETS = 'data/mcdonough_synsets.csv'


if UNIFORM:
    PARAMETERS      = 'intermediate/model_parameters_uniform.pkl'
    RESULTS         = 'results/mcdonough_production_uniform.csv'
else:
    PARAMETERS      = 'intermediate/model_parameters.pkl'
    RESULTS         = 'results/mcdonough_production.csv'

ALSO_ACCEPTABLE = {
        'cat.n.01' : [
            ('kitten.n.01', 'kitten.n.01'),
            ('kitty.n.04', 'kitty.n.04')
            ],
        'dog.n.01' : [
            ('puppy.n.01', 'puppy.n.01')
            ]
        }

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

# Load model parameters from production data.
with open(PARAMETERS, 'rb') as f:
    param     = pickle.load(f)
    h         = torch.tensor(param['h'], device='cuda')

# Prepare optimization.
dist_matrix = torch.tensor(np.load(DIST_MATRIX), device='cuda')
rel_freqs   = torch.tensor(rel_freqs, device='cuda')

# Load experiment stimuli.
booklet = pd.read_csv(BOOKLET)

# Load wordnet -> imagenet dictionary.
booklet_synsets = pd.read_csv(BOOKLET_SYNSETS)
booklet_dic = {}
for i, row in booklet_synsets.iterrows():
    booklet_dic[row['wordnet_synset']] = row['imagenet_synset']

# Assemble input data.
item_wordnet_synsets = []
item_vocab_indices   = []
item_lexicon_indices = []
item_group           = []
all_items            = set()

for i, row in booklet.iterrows():
    for col, group in [
            ('early_high', 'early'),
            ('early_low',  'early'),
            ('late_low',   'late')]:
        wordnet  = row[col]
        imagenet = booklet_dic[wordnet]

        if wordnet in all_items:
            continue
        else:
            all_items.add(wordnet)

        item_wordnet_synsets.append(wordnet)
        item_vocab_indices.append(prod_vocab_lookup[wordnet, imagenet])
        item_lexicon_indices.append(full_lexicon_lookup[wordnet, imagenet])
        item_group.append(group)

items = pd.DataFrame({
    'wordnet_synset' : item_wordnet_synsets,
    'vocab_index'   : item_vocab_indices,
    'lexicon_index' : item_lexicon_indices,
    'group'         : item_group,
    }) 

# Parameters.
kernel_widths = model.compute_kernel_widths(h)
priors = model.rel_freqs_to_priors(rel_freqs, uniform=UNIFORM)

# Results.
data = []

# Production.
dist_submatrix = dist_matrix[:, item_lexicon_indices, :]

prod_ranks, prod_posts, full_post = (
        model.predict_production_ranks_and_posteriors(
            prod_sense_dist       = dist_submatrix,
            identity_prod_indices = item_vocab_indices,
            kernel_widths         = kernel_widths,
            priors                = priors,
            full_dist_matrix      = dist_matrix,
            child_prod_indices    = item_vocab_indices,
            allow_identity_prod   = True,
            target_identity_prod  = True,
            get_full_matrix       = True))

top_prods, top_posts = model.predict_top_k_prods(
        prod_sense_dist       = dist_submatrix,
        identity_prod_indices = item_vocab_indices,
        kernel_widths         = kernel_widths,
        priors                = priors,
        full_dist_matrix      = dist_matrix,
        allow_identity_prod   = True)

for i, post in enumerate(prod_posts):
    prob_correct = post.detach().cpu().numpy()
    if items.loc[i, 'wordnet_synset'] in ALSO_ACCEPTABLE:
        for pair in ALSO_ACCEPTABLE[items.loc[i, 'wordnet_synset']]:
            index = prod_vocab_lookup[pair]
            prob_correct += full_post[index, i].detach().cpu().numpy()

    this_prods = [prod_vocab.loc[t, 'wordnet_synset'] for t in top_prods[i]]
    this_posts = [t.detach().cpu().numpy().round(2) for t in top_posts[i]]

    data.append({
        'group'       : items.loc[i, 'group'],
        'synset'      : items.loc[i, 'wordnet_synset'],
        'probability' : post.detach().cpu().numpy(),
        'response'    : 'correct',
        'top prods'   : this_prods,
        'top posts'   : this_posts,
        })
    data.append({
        'group'       : items.loc[i, 'group'],
        'synset'      : items.loc[i, 'wordnet_synset'],
        'probability' : 1. - post.detach().cpu().numpy(),
        'response'    : 'incorrect',
        'top prods'   : this_prods,
        'top posts'   : this_posts,
        })

data = pd.DataFrame(data)
data['probability'] = data['probability'].astype(float)
data = data.sort_values(by=['group', 'response', 'synset'])
print(data.round(2))

# Plot results.
plt.ion()

plt.figure()
sns.barplot(x='group', hue='response', y='probability', data=data)

# Save results.
data.to_csv(RESULTS, index=False)
