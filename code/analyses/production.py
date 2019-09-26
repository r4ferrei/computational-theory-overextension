import os
import pickle
import argparse

import pandas as pd
import numpy as np
import torch
import torch.utils.data
from torch import optim

import model

INIT_H = 1

OVEREXTENSION = 'intermediate/filtered_overextension_pairs.csv'
PROD_VOCAB    = 'intermediate/filtered_prod_vocab.csv'
FULL_LEXICON  = 'intermediate/filtered_lexicon.csv'
REL_FREQ      = 'intermediate/prod_vocab_rel_freqs.csv'
DIST_MATRIX   = 'intermediate/dist_matrix.npy'

RESULT_TRAIN  = 'results/production_training_result'
RESULT_PRED   = 'results/production_pred_result'

CV_RESULT     = 'results/cv_production_pred_result'

parser = argparse.ArgumentParser()
parser.add_argument('--features', type=str, help="e.g. 0,1 or 2")
parser.add_argument('--pretrained', type=str, help="weights file")
parser.add_argument('--uniform', action='store_true', dest='uniform')
parser.add_argument('--baseline', action='store_true', dest='baseline')
parser.add_argument('--cv', action='store_true')
args = parser.parse_args()

if args.features:
    FEATURES = [int(x) for x in args.features.split(',')]
else:
    FEATURES = [0,1,2]

UNIFORM  = args.uniform
BASELINE = args.baseline
CV       = args.cv

if BASELINE:
    FEATURES_NAME = 'baseline'
else:
    FEATURES_NAME = ','.join([str(x) for x in FEATURES])

if args.pretrained:
    with open(args.pretrained, 'rb') as f:
        PRETRAINED = pickle.load(f)
else:
    PRETRAINED = None

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

# Construct array of relative frequencies aligned with prod_vocab.
rel_freq_df = pd.read_csv(REL_FREQ)
rel_freq_dic = {}
for i, row in rel_freq_df.iterrows():
    rel_freq_dic[row['wordnet_synset']] = row['rel_freq']

rel_freqs = np.zeros((len(prod_vocab),))
for i, row in prod_vocab.iterrows():
    synset = row['wordnet_synset']
    rel_freqs[i] = rel_freq_dic[synset]

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

# Prepare optimization.
dist_matrix = torch.tensor(np.load(DIST_MATRIX), device='cuda')
dist_matrix = dist_matrix[:, :, FEATURES] # sub-features or all

rel_freqs   = torch.tensor(rel_freqs, device='cuda')

if PRETRAINED:
    h = torch.tensor(PRETRAINED['h'], device='cuda', requires_grad=True,
            dtype=dist_matrix.dtype)
else:
    h = torch.tensor([INIT_H], device='cuda', requires_grad=True,
            dtype=dist_matrix.dtype)

class OverextensionDataset(torch.utils.data.Dataset):
    def __init__(self, overextension_indices):
        super().__init__()
        self.indices = overextension_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.indices[i]

def train(dataset, save=True):
    opt = optim.Adam([h], lr = 0.001)

    min_loss = float('inf')
    tol = 1e-6
    patience = 30
    patience_counter = 0
    epoch = 0

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    while patience_counter < patience:
        if BASELINE:
            break

        print("Epoch %d, min loss %.6f, h %.4f" % (epoch+1, min_loss, h))
        epoch += 1

        total_loss = 0

        for batch in dataloader:
            opt.zero_grad()

            child_prod_indices    = batch['child_prod']
            child_sense_indices   = batch['child_sense']
            identity_prod_indices = batch['identity_prod']

            dist_submatrix = dist_matrix[:, child_sense_indices, :]

            kernel_widths = model.compute_kernel_widths(h)
            priors        = model.rel_freqs_to_priors(
                    rel_freqs, uniform=UNIFORM)

            nll = model.production_nll(
                    prod_sense_dist       = dist_submatrix,
                    child_prod_indices    = child_prod_indices,
                    identity_prod_indices = identity_prod_indices,
                    kernel_widths         = kernel_widths,
                    priors                = priors,
                    full_dist_matrix      = dist_matrix)

            nll.backward()
            opt.step()

            total_loss += float(nll.detach().cpu().numpy())

        if total_loss < min_loss:
            min_loss = total_loss
            patience_counter = 0
        else:
            patience_counter += 1

    # Save training result.
    if save:
        train_result = {'h' : h.detach().cpu().numpy()}

        result_train_filename = "{}_{}_{}.pkl".format(
                RESULT_TRAIN,
                'uniform' if UNIFORM else 'frequency',
                FEATURES_NAME)

        with open(result_train_filename, 'wb') as f:
            pickle.dump(train_result, f)

def predict(dataset, overextension_indices=overextension_indices,
        allow_identity_prod=False, target_identity_prod=False,
        save_result=False):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    ranks = np.zeros((0,))
    posts = np.zeros((0,))

    for batch in dataloader:
            child_prod_indices    = batch['child_prod']
            child_sense_indices   = batch['child_sense']
            identity_prod_indices = batch['identity_prod']

            dist_submatrix = dist_matrix[:, child_sense_indices, :]

            kernel_widths = model.compute_kernel_widths(h)
            priors        = model.rel_freqs_to_priors(
                    rel_freqs, uniform=UNIFORM)

            this_ranks, this_posts = model.predict_production_ranks_and_posteriors(
                    prod_sense_dist       = dist_submatrix,
                    identity_prod_indices = identity_prod_indices,
                    kernel_widths         = kernel_widths,
                    priors                = priors,
                    full_dist_matrix      = dist_matrix,
                    child_prod_indices    = child_prod_indices,
                    allow_identity_prod   = allow_identity_prod,
                    target_identity_prod  = target_identity_prod,
                    baseline              = BASELINE)

            ranks = np.append(ranks, this_ranks.detach().cpu().numpy())
            posts = np.append(posts, this_posts.detach().cpu().numpy())

    pred_result = {
            'ranks' : ranks,
            'posts' : posts,
            }

    if save_result:
        result_pred_filename = "{}_{}_{}.pkl".format(
                RESULT_PRED,
                'uniform' if UNIFORM else 'frequency',
                FEATURES_NAME)
        with open(result_pred_filename, 'wb') as f:
            pickle.dump(pred_result, f)

    def prediction_df():
        rows = [
                {
                    'prod'  : prod_vocab.loc[oind['child_prod'], 'wordnet_synset'],
                    'sense' : full_lexicon.loc[oind['child_sense'], 'wordnet_synset'],
                    'rank'  : ranks[i],
                    'post'  : posts[i],
                    }
                for i, oind in enumerate(overextension_indices)
                ]
        df = pd.DataFrame(rows)
        return df[['prod', 'sense', 'post', 'rank']]

    df = prediction_df()
    print(df['rank'].median())
    return df

if CV:
    posts = []
    ranks = []

    for i in range(len(overextension_indices)):
        loo_indices = overextension_indices[:i] + overextension_indices[(i+1):]
        loo_dataset = OverextensionDataset(loo_indices)

        train(loo_dataset, save=False)

        pred_indices = [overextension_indices[i]]
        pred_dataset = OverextensionDataset(pred_indices)

        pred_df = predict(pred_dataset, save_result=False,
                overextension_indices=pred_indices)
        assert(len(pred_df) == 1)

        posts.append(float(pred_df.loc[0, 'post']))
        ranks.append(  int(pred_df.loc[0, 'rank']))

    assert(len(posts) == len(ranks) == len(overextension_indices))

    posts = np.array(posts)
    ranks = np.array(ranks)
    pred_result = {
            'ranks' : ranks,
            'posts' : posts,
            }

    result_pred_filename = "{}_{}_{}.pkl".format(
            CV_RESULT,
            'uniform' if UNIFORM else 'frequency',
            FEATURES_NAME)
    with open(result_pred_filename, 'wb') as f:
        pickle.dump(pred_result, f)
else:
    dataset = OverextensionDataset(overextension_indices)

    if not PRETRAINED:
        train(dataset)

    predict(dataset, save_result=True)
