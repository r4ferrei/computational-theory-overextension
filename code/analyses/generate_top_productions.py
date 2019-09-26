import os
import pickle
import argparse

import pandas as pd
import numpy as np
import torch
import torch.utils.data
from torch import optim

import model

OVEREXTENSION = 'intermediate/filtered_overextension_pairs.csv'
PROD_VOCAB    = 'intermediate/filtered_prod_vocab.csv'
FULL_LEXICON  = 'intermediate/filtered_lexicon.csv'
REL_FREQ      = 'intermediate/prod_vocab_rel_freqs.csv'
DIST_MATRIX   = 'intermediate/dist_matrix.npy'

PARAMETERS    = 'intermediate/model_parameters.pkl'

OUTPUT        = 'results/production_top_predictions.csv'

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

# Construct overextension set as indices into matrix.
overextension_df = pd.read_csv(OVEREXTENSION)
overextension_indices = []
for i, row in overextension_df.iterrows():
    production = row['production_wordnet'], row['production_imagenet']
    sense      = row['sense_wordnet'],      row['sense_imagenet']
    overextension_indices.append({
        'child_prod'     : prod_vocab_lookup[production],
        'child_sense'    : full_lexicon_lookup[sense],
        # Mask None as -1 for PyTorch dataset to work.
        'identity_prod'  : prod_vocab_lookup.get(sense, -1),
        })

# Prepare to run model.
dist_matrix = torch.tensor(np.load(DIST_MATRIX), device='cuda')
rel_freqs   = torch.tensor(rel_freqs, device='cuda')

class OverextensionDataset(torch.utils.data.Dataset):
    def __init__(self, overextension_indices):
        super().__init__()
        self.indices = overextension_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.indices[i]

def predict(dataset, overextension_indices=overextension_indices,
        allow_identity_prod=False, target_identity_prod=False,
        save_result=False, k=5):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    prods = []
    for batch in dataloader:
            child_prod_indices    = batch['child_prod']
            child_sense_indices   = batch['child_sense']
            identity_prod_indices = batch['identity_prod']

            dist_submatrix = dist_matrix[:, child_sense_indices, :]

            kernel_widths = model.compute_kernel_widths(h)
            priors        = model.rel_freqs_to_priors(
                    rel_freqs, uniform=False)

            this_prods, _this_posts = model.predict_top_k_prods(
                    prod_sense_dist       = dist_submatrix,
                    identity_prod_indices = identity_prod_indices,
                    kernel_widths         = kernel_widths,
                    priors                = priors,
                    full_dist_matrix      = dist_matrix,
                    allow_identity_prod   = allow_identity_prod,
                    k                     = k)

            for i, tp in enumerate(this_prods):
                d = ({
                    'production': prod_vocab.loc[
                        int(child_prod_indices[i].detach().cpu()),
                        'wordnet_synset'],
                    'referent': full_lexicon.loc[
                        int(child_sense_indices[i].detach().cpu()),
                        'wordnet_synset'],
                    })

                for j in range(k):
                    d[j+1] = prod_vocab.loc[tp[j], 'wordnet_synset']
                d['correct'] = (
                        int(child_prod_indices[i].detach().cpu())
                        in tp)
                prods.append(d)

    df = pd.DataFrame(prods)
    if save_result:
        df.to_csv(OUTPUT, index=False)
    return df

dataset = OverextensionDataset(overextension_indices)
df = predict(dataset, save_result=True)
