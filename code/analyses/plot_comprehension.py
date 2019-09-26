import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.stats.weightstats as stats

import plotting_style

RESULTS         = 'results/comprehension.csv'

MCDONOUGH_LABEL = 'Empirical data (McDonough, 2002)'
MODEL_LABEL     = 'Model prediction'

PAPER_NUMBERS = {
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

df = pd.read_csv(RESULTS)

model_data = []

for _, row in df.iterrows():
    model_data.append({
        'response'   : MAP_CORRECT_ERROR[row['response']],
        'condition'  : MAP_CONDITION[row['condition']],
        'proportion' : row['probability'],
        })

model_data = pd.DataFrame(model_data)

# Statistical test.

comparisons = [
        ('High contrast (Early)', 'Low contrast (Early)'),
        ('Low contrast (Early)', 'Low contrast (Late)')]
for cond1, cond2 in comparisons:
    data1 = model_data[
            (model_data['response'] == MAP_CORRECT_ERROR['correct']) &
            (model_data['condition'] == MAP_CONDITION[cond1])][
                    'proportion'].values
    data2 = model_data[
            (model_data['response'] == MAP_CORRECT_ERROR['correct']) &
            (model_data['condition'] == MAP_CONDITION[cond2])][
                    'proportion'].values

    t, p, df = stats.ttest_ind(data1, data2, usevar='unequal')

    print('\nConditions: %s, %s' % (cond1, cond2))
    print('n1 = %d, n2 = %d' % (len(data1), len(data2)))
    print('M1 = %.2f, M2 = %.2f' % (np.mean(data1), np.mean(data2)))
    print('t(%d) = %.2f, p = %.6f' % (df, t, p))



# McDonough and plot.

mcdonough_data = pd.DataFrame([
    {
        'response'   : MAP_CORRECT_ERROR['correct'],
        'condition'  : MAP_CONDITION['High contrast (Early)'],
        'proportion' : PAPER_NUMBERS['high_early'],
        },
    {
        'response'   : MAP_CORRECT_ERROR['error'],
        'condition'  : MAP_CONDITION['High contrast (Early)'],
        'proportion' : 1 - PAPER_NUMBERS['high_early'],
        },
    {
        'response'   : MAP_CORRECT_ERROR['correct'],
        'condition'  : MAP_CONDITION['Low contrast (Early)'],
        'proportion' : PAPER_NUMBERS['low_early'],
        },
    {
        'response'   : MAP_CORRECT_ERROR['error'],
        'condition'  : MAP_CONDITION['Low contrast (Early)'],
        'proportion' : 1 - PAPER_NUMBERS['low_early'],
        },
    {
        'response'   : MAP_CORRECT_ERROR['correct'],
        'condition'  : MAP_CONDITION['Low contrast (Late)'],
        'proportion' : PAPER_NUMBERS['low_late'],
        },
    {
        'response'   : MAP_CORRECT_ERROR['error'],
        'condition'  : MAP_CONDITION['Low contrast (Late)'],
        'proportion' : 1 - PAPER_NUMBERS['low_late'],
        },
    ])

plotting_style.setup()

fig, (ax_mcdonough, ax_model) = plt.subplots(
        nrows=1, ncols=2, sharey=True,
        figsize=(17.8 / 2.54, 7.62 / 2.54))

condition_name  = 'Condition'
proportion_name = 'Mean proportion selections'
response_name   = 'Response'
model_data = model_data.rename(columns = {
    'condition'    : condition_name,
    'proportion'   : proportion_name,
    'response'     : response_name,
    })
mcdonough_data = mcdonough_data.rename(columns = {
    'condition'    : condition_name,
    'proportion'   : proportion_name,
    'response'     : response_name,
    })

plt.ion()

def plot_data(data, ax):
    sns.barplot(
            x         = condition_name,
            hue       = response_name,
            y         = proportion_name,
            data      = data,
            order     = [
                MAP_CONDITION['High contrast (Early)'],
                MAP_CONDITION['Low contrast (Early)'],
                MAP_CONDITION['Low contrast (Late)'],
                ],
            hue_order = ['Correct', 'Error'],
            ci        = 95,
            capsize   = 0.1,
            ax        = ax,
            )
    ax.set_xlabel('')
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.legend_.remove()

plot_data(model_data, ax_model)

plot_data(mcdonough_data, ax_mcdonough)

ax_mcdonough.legend(ncol = 2, loc = 'upper right',
        title = 'Referent selection')

ax_model.set_ylabel('')

ax_model.set_title(MODEL_LABEL)
ax_mcdonough.set_title(MCDONOUGH_LABEL)

sns.despine()

plt.tight_layout()

plt.savefig('plots/comprehension.pdf', dpi=1200)#, bbox_inches='tight')

plt.show()
