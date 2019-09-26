import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import plotting_style

PROPORTIONS = 'results/logistic_regression_proportions.csv'
RESULT      = 'plots/logistic_regression_proportions.pdf'

df = pd.read_csv(PROPORTIONS)

plotting_style.setup()

plt.ion()
plt.figure(figsize=(8.7 / 2.54, 8.7 / 2.54))

patches, text, autotext = plt.pie(
        df['proportion'].values,
        autopct    = lambda p: '{p:.0f}\\%'.format(p=p),
        radius     = .7,
        center     = (0, 0),
        startangle = -55)

plt.setp(autotext, size=12, weight='bold', color='w')
plt.legend(patches, df['feature'].values,
        loc='center left', bbox_to_anchor = (.8, 0, 0.4, 1))

def make_examples_text(pairs):
    text = r'\begin{eqnarray*}'
    for i, (prod, sense) in enumerate(pairs):
        text += prod
        text += r'\hspace{-0.5em}'
        text += r'&\rightarrow&'
        text += r'\hspace{-0.5em}'
        text += r'\mathrm{%s}' % sense
        text += r'\\[-4pt]'
    text += r'\end{eqnarray*}'
    return text

assert(list(df['feature'].values) == ['categorical', 'visual', 'predicate'])

text_positions = [(.5, .7), (-1.3, .85), (-.65, -1.4)]
example_lists  = [
        [
            ('dog', 'squirrel'),
            ('cow', 'zebra'),
            ('flower', 'tree'),
            ('airplane', 'submarine')
            ],
        [
            ('apple', 'egg'),
            ('hat', 'bowl'),
            ('ball', 'orange'),
            ('clock', 'telephone'),
            ],
        [
            ('apple', 'orange~juice'),
            ('key', 'door'),
            ('tea', 'teapot'),
            ('spoon', 'fork'),
            ],
        ]

for i, (text_pos, examples) in enumerate(zip(text_positions, example_lists)):
    plt.text(text_pos[0], text_pos[1],
            make_examples_text(examples),
            bbox = {
                'facecolor' : 'none',
                'edgecolor' : patches[i].get_facecolor()})

plt.tight_layout()

plt.savefig(RESULT, dpi=1200)#, bbox_inches='tight')

plt.show()
