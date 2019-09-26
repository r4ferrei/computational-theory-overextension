import random
from pprint import pprint

import pandas as pd
import statsmodels.formula.api as smf
import scipy.stats

import semantic_space

semantic_space.load_precomputed(
        dists = 'intermediate/dist_matrix_square.npy',
        vocab = 'intermediate/filtered_lexicon.csv')

random.seed(2019)

OVEREXTENSION = 'intermediate/filtered_overextension_pairs.csv'

BIC_RESULT         = 'results/logistic_regression_bic.csv'
PROPORTIONS_RESULT = 'results/logistic_regression_proportions.csv'
FEATURES_RESULT    = 'results/logistic_regression_features.csv'

# Load true overextension dataset.
df = pd.read_csv(OVEREXTENSION)
df['is_overextension'] = 1

def make_production(row):
    return (row['production_wordnet'], row['production_imagenet'])

def make_sense(row):
    return (row['sense_wordnet'], row['sense_imagenet'])

overextension_tuples = set([
    (make_production(row), make_sense(row))
    for _, row in df.iterrows()
    ])

# Create negative shuffled dataset.
prods  = [make_production(row) for _, row in df.iterrows()]
senses = [make_sense(row)      for _, row in df.iterrows()]

random.shuffle(senses)

# Fix shuffled pairs that are still true overextension pairs.
while True:
    bad_index = None
    for i in range(len(prods)):
        p = prods[i]
        s = senses[i]
        if (p, s) in overextension_tuples:
            bad_index = i

    if bad_index:
        j = random.randint(0, len(prods)-1)
        senses[bad_index], senses[j] = senses[j], senses[bad_index]
    else:
        break

# Build binary classification dataset.
data = df.to_dict('records')
for p, s in zip(prods, senses):
    data.append({
        'production_wordnet'  : p[0],
        'production_imagenet' : p[1],
        'sense_wordnet'       : s[0],
        'sense_imagenet'      : s[1],
        'is_overextension'    : 0,
        })
df = pd.DataFrame(data)

# Obtain distances in semantic space.
dist_fns = {
        'categorical' : semantic_space.categorical_distance,
        'visual'      : semantic_space.visual_distance,
        'predicate'   : semantic_space.predicate_distance,
        }

def get_pair_for_distance(row, dist_name):
    if dist_name in ['categorical', 'predicate']:
        return row['production_wordnet'], row['sense_wordnet']
    elif dist_name == 'visual':
        return row['production_imagenet'], row['sense_imagenet']
    else:
        assert(False)

for name, fn in dist_fns.items():
    df[name] = [
            fn(*get_pair_for_distance(row, name))
            for _, row in df.iterrows()
            ]

# Feature correlations.

positive = df[df['is_overextension'] == 1]

features = {
        'cat'  : positive['categorical'],
        'vis'  : positive['visual'],
        'pred' : positive['predicate'],
        }

cor_rhos = {}
cor_ps   = {}

for f1 in features:
    for f2 in features:
        rho, p = scipy.stats.spearmanr(features[f1], features[f2])
        cor_rhos[(f1, f2)] = rho
        cor_ps[(f1, f2)] = p

# Print correlations.
for f1, f2 in cor_rhos.keys():
    print('%s,%s: %.3f (p = %.3f)' %
            (f1, f2, cor_rhos[f1, f2], cor_ps[f1, f2]))

# Normalize distances.
for name in dist_fns:
    df[name] = (df[name] - df[name].mean()) / df[name].std()

# Perform logistic regression.
formula = 'is_overextension ~ categorical + visual + predicate'
model = smf.logit(formula=formula, data=df)
result = model.fit()

print(result.summary())

# Measure logit activation feature proportions.
max_counts = {
        'categorical' : 0,
        'visual'      : 0,
        'predicate'   : 0,
        }
features_df = []
for _, row in df.iterrows():
    if not row['is_overextension']: continue

    acts_names = []
    for feature in max_counts:
        acts_names.append((
            result.params[feature] * row[feature],
            feature))
    acts_names = list(reversed(sorted(acts_names)))
    max_counts[acts_names[0][1]] += 1 / len(prods)
    features_df.append({
        'production'  : row['production_wordnet'],
        'sense'       : row['sense_wordnet'],
        'feature'     : acts_names[0][1],
        'activation'  : acts_names[0][0],
        'categorical' : result.params['categorical'] * row['categorical'],
        'visual'      : result.params['visual']      * row['visual'],
        'predicate'   : result.params['predicate']   * row['predicate'],
        })

print('Max activation proportions:')
pprint(max_counts)

proportions = pd.DataFrame({
    'feature'    : list(max_counts.keys()),
    'proportion' : list(max_counts.values())
    })
proportions.to_csv(PROPORTIONS_RESULT, index=False)

features_df = pd.DataFrame(features_df)\
        .sort_values(by=['feature', 'activation'])\
        .round(2)\
        [[
            'production',
            'sense',
            'feature',
            'categorical',
            'visual',
            'predicate',
            ]]

features_df.to_csv(FEATURES_RESULT, index=False)

# Full and partial cross-validated accuracies.
formulas = {
        'full'         : 'is_overextension ~ categorical + visual + predicate',
        'cat + vis'    : 'is_overextension ~ categorical + visual',
        'cat + pred'   : 'is_overextension ~ categorical + predicate',
        'vis + pred'   : 'is_overextension ~ visual + predicate',
        'cat'          : 'is_overextension ~ categorical',
        'vis'          : 'is_overextension ~ visual',
        'pred'         : 'is_overextension ~ predicate',
        }

for i in range(len(df)):
    print('LOOCV %d/%d' % (i+1, len(df)))

    for name, formula in formulas.items():
        df_sub     = df.drop(i)
        model_sub  = smf.logit(formula=formula, data=df_sub)
        result_sub = model_sub.fit()
        pred_sub   = result_sub.predict(df.loc[[i], :])
        pred_class = (pred_sub.values[0] >= .5)
        df.loc[i, name] = pred_class

for name in formulas:
    print('[%s] LOOCV accuracy: %.3f' %
            (name, (df['is_overextension'] == df[name]).mean()))

# Compare BICs.
bic = []
for name, formula in formulas.items():
    model  = smf.logit(formula=formula, data=df)
    result = model.fit()
    bic.append({
        'model'        : name,
        'bic'          : result.bic,
        'loo_acc_mean' : (df['is_overextension'] == df[name]).mean(),
        'loo_acc_se'   : (df['is_overextension'] == df[name]).sem(),
        })

bic = pd.DataFrame(bic).sort_values(by='bic')

print('BIC scores:')
print(bic.round(2))
bic.to_csv(BIC_RESULT, index=False)

# Print correlations.
for f1, f2 in cor_rhos.keys():
    print('%s,%s: %.3f (p = %.3f)' %
            (f1, f2, cor_rhos[f1, f2], cor_ps[f1, f2]))
