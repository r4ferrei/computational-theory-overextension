import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import plotting_style

plotting_style.setup()

BASENAME = 'results/cv_production_pred_result_frequency'

MODELS = {
        'all features': [
            [0,1,2]
            ],
        'categorical + predicate': [
            [0, 2]
            ],
        'categorical + visual': [
            [0, 1]
            ],
        'visual + predicate': [
            [1, 2]
            ],
        'categorical': [
            [0],
            ],
        'visual': [
            [1],
            ],
        'predicate': [
            [2]
            ],
        'baseline': [
            []
            ],
        }

STYLES = {
        'categorical + visual': {
            'linestyle' : '-.',
            },
        'categorical + predicate': {
            'linestyle' : '-.',
            },
        'visual + predicate': {
            'linestyle' : '-.',
            },
        'categorical': {
            'linestyle' : '--',
            },
        'visual': {
            'linestyle' : '--',
            },
        'predicate': {
            'linestyle' : '--',
            },
        'baseline': {
            'color'     : 'grey',
            'linestyle' : ':',
            }
        }

def make_name(model):
    if model:
        return ','.join(str(x) for x in model)
    else:
        return 'baseline'

MAX_X = 40


plt.ion()
plt.figure(figsize=(17.8 / 2.54, 10.16 / 2.54))

for name, models in MODELS.items():
    ranks = []
    for model in models:
        fname = "{}_{}.pkl".format(BASENAME, make_name(model))
        with open(fname, 'rb') as f:
            res = pickle.load(f)
        ranks.append(res['ranks'])

    accs = []
    for i in range(1, MAX_X+1):
        accs.append(np.mean(
            [(r <= i).mean() for r in ranks]))

    plt.plot(list(range(1, MAX_X+1)), accs,
            **STYLES.get(name, {}))

#plt.yticks(np.arange(0, 1.1, .2))

plt.legend(list(MODELS.keys()), ncol=4,
        loc='lower center',
        bbox_to_anchor = (.5, plt.ylim()[1] + .05))

plt.xlabel('$m$ (number of choices)')
plt.ylabel('Reconstruction accuracy')

plt.tight_layout()
plt.savefig('plots/individual_production_curves.pdf',
        dpi=1200)#, bbox_inches='tight')

plt.show()
