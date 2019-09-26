import pickle

import pandas as pd
import numpy as np

FREQUENCY_BASENAME = 'results/production_pred_result_frequency'
UNIFORM_BASENAME   = 'results/production_pred_result_uniform'

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

def make_filename(basename, model):
    return '%s_%s.pkl' % (basename, model)

def compute_bic(model_name, posts):
    loglik = np.sum(np.log(posts))
    k      = NUM_PARAMS[model_name]
    n      = len(posts)
    return np.log(n)*k - 2*loglik

data = []
for model_name, model in MODELS.items():
    with open(make_filename(FREQUENCY_BASENAME, model), 'rb') as f:
        frequency = pickle.load(f)
    with open(make_filename(UNIFORM_BASENAME, model), 'rb') as f:
        uniform = pickle.load(f)

    frequency_bic = compute_bic(model_name, frequency['posts'])
    uniform_bic   = compute_bic(model_name, uniform['posts'])

    data.append({
        'model'         : model_name,
        'frequency bic' : frequency_bic,
        'uniform bic'   : uniform_bic,
        })

data = pd.DataFrame(data)[['model', 'frequency bic', 'uniform bic']]
data.to_csv('results/production_bics.csv', index=False)
