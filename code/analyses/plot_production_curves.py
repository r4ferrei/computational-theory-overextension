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
        'feature pairs': [
            [0,1], [0,2], [1,2]
            ],
        'single features': [
            [0], [1], [2]
            ],
        'baseline': [
            []
            ],
        }

STYLES = {
        'feature pairs': {
            'linestyle' : '-.',
            },
        'single features': {
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
plt.figure(figsize=(8.7 / 2.54, 7.62 / 2.54))
#plt.figure(figsize=(4.46, 3.9))

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

plt.legend(list(MODELS.keys()), ncol=2,
        loc='lower center',
        bbox_to_anchor = (.5, plt.ylim()[1] + .05))

plt.xlabel('$m$ (number of choices)')
plt.ylabel('Reconstruction accuracy')

sns.despine()

plt.tight_layout()
plt.savefig('plots/production_curves.pdf', dpi=1200)#, bbox_inches='tight')

plt.show()
