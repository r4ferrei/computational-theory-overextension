import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.stats.weightstats as stats

import plotting_style

parser = argparse.ArgumentParser()
parser.add_argument('--freq_prior', action='store_true', dest='freq_prior')
parser.add_argument('--filtered', action='store_true', dest='filtered')
args = parser.parse_args()

FREQ_PRIOR = args.freq_prior
FILTERED   = args.filtered

if FREQ_PRIOR:
    assert(not FILTERED)
    COMP_RESULTS = 'results/comprehension_freq_prior.csv'
    OUTPUT       = 'plots/asymmetry_freq_prior.pdf'
elif FILTERED:
    COMP_RESULTS = 'results/comprehension_filtered.csv'
    OUTPUT       = 'plots/asymmetry_filtered.pdf'
else:
    COMP_RESULTS = 'results/comprehension.csv'
    OUTPUT       = 'plots/asymmetry.pdf'

MCDONOUGH_LABEL = 'Empirical data (McDonough, 2002)'
MODEL_LABEL     = 'Model prediction'

COMP_PAPER_NUMBERS = {
        'high_early' : 0.94,
        'low_early'  : 0.68,
        'low_late'   : 0.72,
        }

MAP_CORRECT_ERROR = {
        'correct' : 'Correct',
        'error'   : 'Error',
        }

MAP_CONDITION = {
        'High contrast (Early)' : 'High contrast\n (Early nouns)',
        'Low contrast (Early)'  : 'Low contrast\n (Early nouns)',
        'Low contrast (Late)'   : 'Low contrast\n (Late nouns)',
        }

if FILTERED:
    PROD_RESULT = 'results/mcdonough_production_filtered.csv'
else:
    PROD_RESULT = 'results/mcdonough_production.csv'

PROD_PAPER_NUMBERS = {
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

# Production.

prod_df = pd.read_csv(PROD_RESULT)
prod_data = []

for _, row in prod_df.iterrows():
    if row['response'] == 'correct':
        prod_data.append({
            'model_group' : MODEL_LABEL,
            'word_group'  : MAP_EARLY_LATE[row['group']],
            'proportion'  : row['probability'],
            })

prod_data.append({
    'model_group' : MCDONOUGH_LABEL,
    'word_group'  : MAP_EARLY_LATE['early'],
    'proportion'  : PROD_PAPER_NUMBERS['early']['correct'],
    })
prod_data.append({
    'model_group' : MCDONOUGH_LABEL,
    'word_group'  : MAP_EARLY_LATE['late'],
    'proportion'  : PROD_PAPER_NUMBERS['late']['correct'],
    })

prod_data = pd.DataFrame(prod_data)

# Comprehension.

comp_df = pd.read_csv(COMP_RESULTS)

comp_model_data = []

for _, row in comp_df.iterrows():
    comp_model_data.append({
        'response'   : MAP_CORRECT_ERROR[row['response']],
        'condition'  : MAP_CONDITION[row['condition']],
        'proportion' : row['probability'],
        })

comp_model_data = pd.DataFrame(comp_model_data)

comp_mcdonough_data = pd.DataFrame([
    {
        'response'   : MAP_CORRECT_ERROR['correct'],
        'condition'  : MAP_CONDITION['High contrast (Early)'],
        'proportion' : COMP_PAPER_NUMBERS['high_early'],
        },
    {
        'response'   : MAP_CORRECT_ERROR['error'],
        'condition'  : MAP_CONDITION['High contrast (Early)'],
        'proportion' : 1 - COMP_PAPER_NUMBERS['high_early'],
        },
    {
        'response'   : MAP_CORRECT_ERROR['correct'],
        'condition'  : MAP_CONDITION['Low contrast (Early)'],
        'proportion' : COMP_PAPER_NUMBERS['low_early'],
        },
    {
        'response'   : MAP_CORRECT_ERROR['error'],
        'condition'  : MAP_CONDITION['Low contrast (Early)'],
        'proportion' : 1 - COMP_PAPER_NUMBERS['low_early'],
        },
    {
        'response'   : MAP_CORRECT_ERROR['correct'],
        'condition'  : MAP_CONDITION['Low contrast (Late)'],
        'proportion' : COMP_PAPER_NUMBERS['low_late'],
        },
    {
        'response'   : MAP_CORRECT_ERROR['error'],
        'condition'  : MAP_CONDITION['Low contrast (Late)'],
        'proportion' : 1 - COMP_PAPER_NUMBERS['low_late'],
        },
    ])

# Statistical test.
# TODO

# Plot.
plotting_style.setup()
plt.ion()

def plot(comp_data, prod_data, ax, ci):
    plot_data = []

    performance_name = 'Performance (proportion correct response)'
    task_name        = 'Task'
    word_group_name  = 'Word group'

    comprehension_high_name = 'Comprehension\n(High contrast)'
    comprehension_low_name  = 'Comprehension\n(Low contrast)'
    production_name          = 'Production'

    # Comprehension.
    for _, row in comp_data.iterrows():
        if 'Correct' not in row['response']: continue

        if 'Early' in row['condition']:
            group = MAP_EARLY_LATE['early']
        else:
            group = MAP_EARLY_LATE['late']

        if 'Low' in row['condition']:
            plot_data.append({
                task_name        : comprehension_low_name,
                word_group_name  : group,
                performance_name : row['proportion'],
                })
        else:
            assert('High' in row['condition'])
            plot_data.append({
                task_name        : comprehension_high_name,
                word_group_name  : group,
                performance_name : row['proportion']
                })

    # Production.
    for _, row in prod_data.iterrows():
        plot_data.append({
            task_name        : 'Production',
            word_group_name  : row['word_group'],
            performance_name : row['proportion'],
            })

    plot_data = pd.DataFrame(plot_data)
    bars = sns.barplot(
            x         = task_name,
            hue       = word_group_name,
            y         = performance_name,
            data      = plot_data,
            hue_order = [MAP_EARLY_LATE['early'], MAP_EARLY_LATE['late']],
            order     = [
                comprehension_high_name,
                comprehension_low_name,
                production_name],
            ci        = ci,
            capsize   = 0.1,
            ax        = ax)

    # Center first bar because there is no high-contrast, late noun condition.
    patches = bars.patches
    shift   = .5 * patches[0].get_width()
    patches[0].set_x(patches[0].get_x() + shift)
    if ci is not None:
        for i in range(3):
            line  = ax.lines[i]
            xdata = line.get_xdata()
            for j in range(len(xdata)):
                xdata[j] += shift
            line.set_xdata(xdata)

    ax.set_xlabel('')
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.1, .2))
    ax.legend_.remove()

fig, (ax_mcdonough, ax_model) = plt.subplots(
        nrows=1, ncols=2, sharey=True,
        figsize=(17.8 / 2.54, 7.62 / 2.54))

plot(
        comp_mcdonough_data, 
        prod_data[prod_data['model_group'] == MCDONOUGH_LABEL],
        ax_mcdonough,
        ci = None)
plot(
        comp_model_data,
        prod_data[prod_data['model_group'] == MODEL_LABEL],
        ax_model,
        ci = 95)

ax_mcdonough.legend(ncol = 1, loc = 'upper right')

ax_model.set_ylabel('')

ax_mcdonough.set_title(MCDONOUGH_LABEL)
ax_model.set_title(MODEL_LABEL)

sns.despine()
plt.tight_layout()

plt.savefig(OUTPUT, dpi=1200)#, bbox_inches='tight')

plt.show()
