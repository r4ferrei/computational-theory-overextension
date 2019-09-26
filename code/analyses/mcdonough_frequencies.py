import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import plotting_style

REL_FREQ = 'intermediate/prod_vocab_rel_freqs.csv'
BOOKLET  = 'intermediate/filtered_booklets.csv'

rel_freq_df = pd.read_csv(REL_FREQ)
rel_freq_dic = {}
for i, row in rel_freq_df.iterrows():
    rel_freq_dic[row['wordnet_synset']] = row['rel_freq']

booklet = pd.read_csv(BOOKLET)

early_freqs = {}
late_freqs  = {}

for i, row in booklet.iterrows():
    early1 = row['early_high']
    early2 = row['early_low']
    late   = row['late_low']

    early_freqs[early1] = rel_freq_dic[early1]
    early_freqs[early2] = rel_freq_dic[early2]
    late_freqs[late]    = rel_freq_dic[late]

data = []
for item, freq in early_freqs.items():
    data.append({
        'type' : 'Early nouns',
        'item' : item,
        'freq' : freq
        })
for item, freq in late_freqs.items():
    data.append({
        'type' : 'Late nouns',
        'item' : item,
        'freq' : freq
        })
data = pd.DataFrame(data)

plotting_style.setup()

plt.ion()
plt.figure(figsize=(8.7 / 2.54, 6.604 / 2.54))

sns.barplot(x='type', y='freq', data=data, capsize=.2)

plt.xlabel('')
plt.ylabel('Frequency in child-directed speech')

plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

plt.tight_layout()

plt.savefig('plots/mcdonough_frequencies.pdf', dpi=1200)#, bbox_inches='tight')
