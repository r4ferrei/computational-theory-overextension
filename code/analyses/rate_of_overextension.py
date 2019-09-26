import pickle
import argparse

import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import model
import plotting_style

REL_FREQ      = 'intermediate/prod_vocab_rel_freqs.csv'
PROD_VOCAB    = 'intermediate/filtered_lexicon.csv' # attention!
FULL_LEXICON  = 'intermediate/filtered_lexicon.csv'
DIST_MATRIX   = 'intermediate/dist_matrix_square.npy'

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

# Prepare optimization.
dist_matrix = torch.tensor(np.load(DIST_MATRIX), device='cuda')
rel_freqs   = torch.tensor(rel_freqs, device='cuda')

# Loop over h parameters and compute overextension rates.

with open('intermediate/model_parameters.pkl', 'rb') as f:
    param = pickle.load(f)
    learned_h = float(param['h'])

MAX_H  = learned_h * 1.55
MIN_H  = 0.13
NUM_HS = 20

data = []
for h_val in np.linspace(MAX_H, MIN_H, num=NUM_HS):
    # Parameters.
    h             = torch.tensor([h_val], dtype=rel_freqs.dtype, device='cuda')
    kernel_widths = model.compute_kernel_widths(h)
    priors        = model.rel_freqs_to_priors(rel_freqs, uniform=False)

    identity_prod_indices = []
    for i in range(len(full_lexicon)):
        wordnet  = full_lexicon.loc[i, 'wordnet_synset']
        imagenet = full_lexicon.loc[i, 'imagenet_synset']
        identity_prod_indices.append(
                prod_vocab_lookup.get((wordnet, imagenet), -1))

    # Prediction.
    ranks, posts, full = model.predict_production_ranks_and_posteriors(
            prod_sense_dist       = dist_matrix,
            identity_prod_indices = identity_prod_indices,
            kernel_widths         = kernel_widths,
            priors                = priors,
            full_dist_matrix      = None,
            child_prod_indices    = None,
            allow_identity_prod   = True,
            target_identity_prod  = True,
            get_full_matrix       = True)

    is_overextended = [False] * len(prod_vocab)
    for i in range(len(full_lexicon)):
        if ranks[i] > 1:
            prod_ind = int(full[:,i].argmax().detach().cpu())
            is_overextended[prod_ind] = True

    for i in range(len(prod_vocab)):
        data.append({
            'h'            : h_val,
            'word'         : prod_vocab.loc[i, 'wordnet_synset'],
            'overextended' : is_overextended[i],
            })

data = pd.DataFrame(data)

# Plot results.
plotting_style.setup()
plt.ion()

plt.figure(figsize=(7.01, 2.6))

sns.lineplot(x='h', y='overextended', data=data)

plt.xlim(MAX_H, MIN_H) # invert axis

plt.ylabel('Predicted rate of overextension')
plt.xlabel('Kernel width parameter')

plt.axvline(x=learned_h, linestyle='dashed', color='grey')

plt.annotate('learned\nparameter',
        xy = (learned_h, 0), xytext = (0.55, 0.05),
        arrowprops = {'arrowstyle': '->'})

plt.tight_layout()

plt.savefig('plots/rate_of_overextension.pdf', dpi=1200, bbox_inches='tight')
plt.savefig('plots/rate_of_overextension.svg', dpi=1200, bbox_inches='tight')

plt.show()
