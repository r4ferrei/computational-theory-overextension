import scipy.io as sio
import os

from nltk.corpus import wordnet as wn

SWOW_DIR = 'data/swow'

class SWOW():
    '''
    Class responsible for computing distance metrics based on the
    English SWOW data.
    
    Computes [0, 1] distance based from cosine similarity based on the R1
    random walk measure (see De Deyne 2018 for details).
    '''

    def __init__(self, dir=SWOW_DIR, labels_filename='labels.txt',
                 sim_filename='random_walk_cosine_similarity.mat'):
        with open(os.path.join(dir, labels_filename), 'r') as f:
            words = [s.strip().lower() for s in f]
            self.labels = dict(zip(words, range(len(words))))

        self.metric = sio.loadmat(os.path.join(dir, sim_filename))
        self.metric = self.metric['S'][0,0][0]

        assert(self.metric.shape == (12176, 12176))

    def distance(self, syn1, syn2):
        if isinstance(syn1, str):
            syn1 = wn.synset(syn1)
        if isinstance(syn2, str):
            syn2 = wn.synset(syn2)

        def find_words(synset):
            res = []
            for name in synset.lemma_names():
                for repl in ['', ' ']:
                    query = name.replace('_', repl).lower()
                    if query in self.labels:
                        res.append(query)
            return res

        words1 = find_words(syn1)
        words2 = find_words(syn2)

        if not words1 or not words2:
            print("Warning: no distance for {}, {}".format(syn1, syn2))
            return None

        # Shortest distance between possible lemmas, converting
        # similarity to distance.
        dists = [1 - self.metric[self.labels[w1], self.labels[w2]]
                for w1 in words1 for w2 in words2]
        return max(0, min(dists)) # correct numerical error
