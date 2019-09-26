import pandas as pd
import numpy as np

np.random.seed(123)


INPUT  = 'results/production_top_predictions.csv'
OUTPUT = 'results/sample_top_productions.csv'

df = pd.read_csv(INPUT)
good = df[df['correct']]
bad  = df[~df['correct']]

good = good.sample(10).drop(columns='correct')
bad  = bad.sample(10).drop(columns='correct')
res  = pd.concat([good, bad], ignore_index=True)

res.to_csv(OUTPUT, index=False)
