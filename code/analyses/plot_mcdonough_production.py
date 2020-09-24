import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.stats.weightstats as stats

import plotting_style

parser = argparse.ArgumentParser()
parser.add_argument('--filtered', action='store_true', dest='filtered')
args = parser.parse_args()

FILTERED = args.filtered

if FILTERED:
    RESULT_FREQ    = 'results/mcdonough_production_filtered.csv'
    RESULT_UNIFORM = 'results/mcdonough_production_uniform_filtered.csv'
    OUTPUT         = 'plots/mcdonough_production_filtered.pdf'
else:
    RESULT_FREQ    = 'results/mcdonough_production.csv'
    RESULT_UNIFORM = 'results/mcdonough_production_uniform.csv'
    OUTPUT         = 'plots/mcdonough_production.pdf'

MCDONOUGH_LABEL = 'McDonough\n(2002)'
FREQUENCY_LABEL = 'Model\n(frequency prior)'
UNIFORM_LABEL   = 'Model\n(uniform prior)'

PAPER_NUMBERS = {
        'early' : {
            'correct'       : 0.69,
            'overextension' : 0.16,
            'other'         : 0.06,
            },
        'late' : {
            'correct'       : 0.07,
            'overextension' : 0.29,
            'other'         : 0.05,
            }
        }

MAP_EARLY_LATE = {
        'early' : 'Early nouns',
        'late'  : 'Late nouns',
        }

df_freq    = pd.read_csv(RESULT_FREQ)
df_uniform = pd.read_csv(RESULT_UNIFORM)

data = []

for _, row in df_freq.iterrows():
    if row['response'] == 'correct':
        data.append({
            'model_group' : FREQUENCY_LABEL,
            'word_group'  : MAP_EARLY_LATE[row['group']],
            'proportion'  : row['probability'],
            })

for _, row in df_uniform.iterrows():
    if row['response'] == 'correct':
        data.append({
            'model_group' : UNIFORM_LABEL,
            'word_group'  : MAP_EARLY_LATE[row['group']],
            'proportion'  : row['probability'],
            })

data.append({
    'model_group' : MCDONOUGH_LABEL,
    'word_group'  : MAP_EARLY_LATE['early'],
    'proportion'  : PAPER_NUMBERS['early']['correct'],
    })
data.append({
    'model_group' : MCDONOUGH_LABEL,
    'word_group'  : MAP_EARLY_LATE['late'],
    'proportion'  : PAPER_NUMBERS['late']['correct'],
    })

data = pd.DataFrame(data)

# Statistical test.

for model_type in [FREQUENCY_LABEL, UNIFORM_LABEL]:
    print('\nModel type: %s' % model_type)

    early = data[
            (data['model_group'] == model_type) &
            (data['word_group']  == MAP_EARLY_LATE['early'])][
                    'proportion'].values
    late = data[
            (data['model_group'] == model_type) &
            (data['word_group']  == MAP_EARLY_LATE['late'])][
                    'proportion'].values

    t, p, df = stats.ttest_ind(early, late, usevar='unequal')

    print('n_early = %d, n_late = %d' % (len(early), len(late)))
    print('M(early) = %.2f, M(late) = %.2f' % (np.mean(early), np.mean(late)))
    print('t(%d) = %.4f, p = %.6f' % (df, t, p))

# Plot.

plotting_style.setup()

fig, ax = plt.subplots(figsize=(8.7 / 2.54, 7.62 / 2.54))

word_group_name = 'Word group'
proportion_name = 'Mean proportion correct labels produced'
data = data.rename(columns = {
    'word_group'   : word_group_name,
    'proportion'   : proportion_name,
    })

plt.ion()

sns.barplot(
        x         = 'model_group',
        hue       = word_group_name,
        y         = proportion_name,
        data      = data,
        order     = [MCDONOUGH_LABEL, FREQUENCY_LABEL, UNIFORM_LABEL],
        hue_order = [MAP_EARLY_LATE['early'], MAP_EARLY_LATE['late']],
        ci        = 95,
        capsize   = 0.1,
        ax        = ax,
        )

ax.set_xlabel('')
ax.set_ylim(0, 0.95)

ax.set_yticks(np.arange(0, 0.9, 0.2))

sns.despine()

plt.tight_layout()

plt.savefig(OUTPUT, dpi=1200)#, bbox_inches='tight')

plt.show()
