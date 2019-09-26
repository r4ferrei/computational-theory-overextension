import os
import random

from nltk.corpus import wordnet as wn
from PIL import Image
import pandas as pd

IMAGENET = 'data/imagenet/'
INPUT    = 'data/overextension_production_nouns.csv'
OUTPUT   = 'overextension_production_nouns_new.csv'

df = pd.read_csv(INPUT).fillna('')
df = df[df['noun_to_noun'] == 1]

def query_user(word):
    word = word.replace(' ', '_')

    print()
    print('Word: %s' % word)

    print()
    print('1. WordNet synset')

    synsets = wn.synsets(word)

    print()
    print('%d synsets:' % len(synsets))
    print()

    for i, synset in enumerate(synsets):
        print('%d: %s' % (i, synset.definition()))

    print()

    while True:
        wordnet_synset = input('Choose WordNet synset [-1 for alternative]: ')

        if wordnet_synset != '-1':
            break

        word = input('Alternative word: ').replace(' ', '_')
        synsets = wn.synsets(word)
        for i, synset in enumerate(synsets):
            print('%d: %s' % (i, synset.definition()))

    wordnet_synset = synsets[int(wordnet_synset)]

    print()
    print('2. ImageNet synset')

    while True:
        index = input('Show image from (0-%d) [-1 for alternative word]: ' %
                (len(synsets)-1))
        if not index: break
        index = int(index)

        if index == -1:
            word = input('Alternative word: ').replace(' ', '_')
            synsets = wn.synsets(word)
            for i, synset in enumerate(synsets):
                print('%d: %s' % (i, synset.definition()))
            continue

        offset = synsets[index].offset()
        imagenet_id = 'n%08d' % offset

        try:
            images = os.listdir(os.path.join(IMAGENET, imagenet_id))
        except FileNotFoundError:
            print('no images')
            continue

        filename = random.choice(images)

        img = Image.open(os.path.join(IMAGENET, imagenet_id, filename))
        img.show()

    print()
    imagenet_synset = input('Choose ImageNet synset: ')
    imagenet_synset = synsets[int(imagenet_synset)]

    return wordnet_synset.name(), imagenet_synset.name()

cache = {}

def try_fill(i, production_or_sense):
    word         = df.loc[i, 'norm_%s' % production_or_sense]
    wordnet_col  = '%s_wordnet'  % production_or_sense
    imagenet_col = '%s_imagenet' % production_or_sense

    if df.loc[i, wordnet_col] and df.loc[i, imagenet_col]:
        return

    try:
        if word not in cache:
            cache[word] = query_user(word)

        wordnet, imagenet = cache[word]
        df.loc[i, wordnet_col]  = wordnet
        df.loc[i, imagenet_col] = imagenet
    except:
        while True:
            mark_as_none = input('Mark as NONE? [y/n] ')
            if mark_as_none in ['y', 'n']:
                break

        if mark_as_none == 'y':
            df.loc[i, wordnet_col] = df.loc[i, imagenet_col] = 'NONE'
            cache[word] = 'NONE', 'NONE'

# Populate cache.
for i in range(len(df)):
    for production_or_sense in ['production', 'sense']:
        word = df.loc[i, 'norm_%s' % production_or_sense]
        wordnet_col  = '%s_wordnet'  % production_or_sense
        imagenet_col = '%s_imagenet' % production_or_sense
        if df.loc[i, wordnet_col] and df.loc[i, imagenet_col]:
            cache[word] = df.loc[i, wordnet_col], df.loc[i, imagenet_col]

for i in range(len(df)):
    try_fill(i, 'production')
    try_fill(i, 'sense')

df.to_csv(OUTPUT, index=False)
