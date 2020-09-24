import pickle

import pandas as pd
import numpy as np

FREQUENCY_BASENAME = 'results/production_pred_result_frequency'
UNIFORM_BASENAME   = 'results/production_pred_result_uniform'

MULTIPLE_WIDTHS_SUFFIX = '_multiple'

MODELS = {
        'baseline'     : 'baseline',
        'categorical'  : '0',
        'visual'       : '1',
        'predicate'    : '2',
        'cat. + vis.'  : '0,1',
        'cat. + pred.' : '0,2',
        'vis. + pred.' : '1,2',
        'all features' : '0,1,2',
        }

# Non-baseline models have a single parameter h, the kernel width.
NUM_PARAMS = {
        'baseline'     : 0,
        'categorical'  : 1,
        'visual'       : 1,
        'predicate'    : 1,
        'cat. + vis.'  : 1,
        'cat. + pred.' : 1,
        'vis. + pred.' : 1,
        'all features' : 1,
        }

# Models with multiple kernel widths.
NUM_PARAMS_MULTIPLE = {
        'baseline'     : 0,
        'categorical'  : 1,
        'visual'       : 1,
        'predicate'    : 1,
        'cat. + vis.'  : 2,
        'cat. + pred.' : 2,
        'vis. + pred.' : 2,
        'all features' : 3,
        }

def make_filename(basename, model, suffix=''):
    return '%s_%s%s.pkl' % (basename, model, suffix)

def compute_bic(model_name, posts, multiple_widths=False):
    loglik = np.sum(np.log(posts))

    if multiple_widths:
        k = NUM_PARAMS_MULTIPLE[model_name]
    else:
        k = NUM_PARAMS[model_name]

    n      = len(posts)
    return np.log(n)*k - 2*loglik

data = []
for model_name, model in MODELS.items():
    with open(make_filename(UNIFORM_BASENAME, model), 'rb') as f:
        uniform = pickle.load(f)
    with open(make_filename(FREQUENCY_BASENAME, model), 'rb') as f:
        frequency = pickle.load(f)
    with open(make_filename(UNIFORM_BASENAME, model,
              MULTIPLE_WIDTHS_SUFFIX if model_name != 'baseline' else ''), 'rb') as f:
        multiple_uniform = pickle.load(f)
    with open(make_filename(FREQUENCY_BASENAME, model,
              MULTIPLE_WIDTHS_SUFFIX if model_name != 'baseline' else ''), 'rb') as f:
        multiple_frequency = pickle.load(f)

    uniform_bic   = compute_bic(model_name, uniform['posts'])
    frequency_bic = compute_bic(model_name, frequency['posts'])
    multiple_uniform_bic  = compute_bic(model_name, multiple_uniform['posts'], multiple_widths=True)
    multiple_frequency_bic  = compute_bic(model_name, multiple_frequency['posts'], multiple_widths=True)

    data.append({
        'model'                   : model_name,
        'uniform bic'             : uniform_bic,
        'frequency bic'           : frequency_bic,
        'multiple uniform bic'    : multiple_uniform_bic,
        'multiple frequency bic'  : multiple_frequency_bic,
        })

data = pd.DataFrame(data)[[
    'model',
    'uniform bic',
    'frequency bic',
    'multiple uniform bic',
    'multiple frequency bic'
    ]]
data.to_csv('results/production_bics.csv', index=False)
