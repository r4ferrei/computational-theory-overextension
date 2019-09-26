import pickle

import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import model
import semantic_space

PARAMETERS = 'intermediate/model_parameters.pkl'
REL_FREQ   = 'intermediate/prod_vocab_rel_freqs.csv'
BOOKLET    = 'intermediate/filtered_booklets.csv'
SYNSETS    = 'data/mcdonough_synsets.csv'

RESULTS    = 'results/comprehension.csv'

semantic_space.load_precomputed(
        dists = 'intermediate/dist_matrix_square_mcdonough.npy',
        vocab = 'intermediate/filtered_lexicon_mcdonough.csv')

# Load frequencies.
rel_freq_df = pd.read_csv(REL_FREQ)
rel_freq_dic = {}
for i, row in rel_freq_df.iterrows():
    rel_freq_dic[row['wordnet_synset']] = row['rel_freq']

# Load stimuli.
booklet = pd.read_csv(BOOKLET)

wordnet_to_imagenet = {}
synset_df = pd.read_csv(SYNSETS)
for i, row in synset_df.iterrows():
    wordnet_to_imagenet[row['wordnet_synset']] = row['imagenet_synset']

# Load model parameters from production data.
with open(PARAMETERS, 'rb') as f:
    param = pickle.load(f)
    h     = torch.tensor(param['h'], device='cuda')

# For each row, construct 3x3 likelihood matrix and measure diagonal.
data = []
data_low = []
for _, row in booklet.iterrows():
    items =  [row['early_high'], row['early_low'], row['late_low']]
    rel_freqs = np.array([rel_freq_dic[it] for it in items])
    rel_freqs = torch.tensor(rel_freqs, device='cuda')

    kernel_widths = model.compute_kernel_widths(h)

    dist_matrix = np.zeros((3,3,3))
    for i, it1 in enumerate(items):
        for j, it2 in enumerate(items):
            dist_matrix[i, j, 0] = semantic_space.categorical_distance(it1, it2)
            dist_matrix[i, j, 1] = semantic_space.visual_distance(
                    wordnet_to_imagenet[it1], wordnet_to_imagenet[it2])
            dist_matrix[i, j, 2] = semantic_space.predicate_distance(it1, it2)
    dist_matrix = torch.tensor(dist_matrix, device='cuda')

    lik  = model.compute_likelihood(dist_matrix, kernel_widths)
    prob = torch.div(lik,
            torch.reshape(torch.sum(lik, dim=1), (-1,1)))
    prob = prob.detach().cpu().numpy()

    data.append({
        'condition'   : 'High contrast (Early)',
        'probability' : prob[0,0],
        'response'    : 'correct',
        })
    data.append({
        'condition'   : 'High contrast (Early)',
        'probability' : 1 - prob[0,0],
        'response'    : 'error',
        })

    data.append({
        'condition'   : 'Low contrast (Early)',
        'probability' : prob[1,1],
        'response'    : 'correct',
        })
    data.append({
        'condition'   : 'Low contrast (Early)',
        'probability' : 1 - prob[1,1],
        'response'    : 'error',
        })

    data.append({
        'condition'   : 'Low contrast (Late)',
        'probability' : prob[2,2],
        'response'    : 'correct',
        })
    data.append({
        'condition'   : 'Low contrast (Late)',
        'probability' : 1 - prob[2,2],
        'response'    : 'error',
        })

    data_low.append({
        'probability' : prob[1,2],
        'response'    : 'overextension',
        })
    data_low.append({
        'probability' : prob[1,0],
        'response'    : 'unrelated',
        })
    data_low.append({
        'probability' : prob[2,1],
        'response'    : 'overextension',
        })
    data_low.append({
        'probability' : prob[2,0],
        'response'    : 'unrelated',
        })

data = pd.DataFrame(data)
data_low = pd.DataFrame(data_low)

# Plots
plt.ion()

# Plot probabilities in High, Low (early) and Low (late) conditions.
plt.figure()
sns.barplot(x='condition', y='probability', hue='response', data=data)

# Plot probability of overextension versus other error in low contrast.
plt.figure()
sns.barplot(x='response', y='probability', data=data_low)

# Save results.
data.to_csv(RESULTS, index=False)
