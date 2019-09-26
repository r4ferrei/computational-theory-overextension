import pandas as pd

OUTPUT = 'tables/overextension.tex'

SOURCES = {
        'barrett_1978'  : 'barrett1978lexical',
        'clark_1973'    : 'clark1973s',
        'fremgen_1980'  : 'fremgen1980overextensions',
        'gruendel_1977' : 'gruendel1977referential',
        'rescorla_1976' : 'rescorla1976concept',
        'rescorla_1980' : 'rescorla1980overextension',
        'rescorla_1981' : 'rescorla_category_1981',
        'thomson_1977'  : 'thomson_who_1977',
        }

df = pd.read_csv('intermediate/filtered_overextension_pairs.csv')
df = df[['production_word', 'production_wordnet', 'sense_word',
    'sense_wordnet', 'source_id']]

df['source_id'] = df['source_id'].apply(lambda s: r'\citet{%s}' % SOURCES[s])

df = df.rename(columns = {
    'production_word'    : r'\bfseries Production',
    'production_wordnet' : r'\bfseries Synset',
    'sense_word'         : r'\bfseries Referent',
    'sense_wordnet'      : r'\bfseries Synset',
    'source_id'          : r'\bfseries Source',
    })

df.to_latex(OUTPUT, index=False, escape=False)
