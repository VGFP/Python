import os
from keras.utils import get_file
import gensim
import patoolib
import subprocess
import numpy as np
import requests
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
figsize(10, 10)
import random
from sklearn.manifold import TSNE
import json
from collections import Counter
import csv
#import geopandas as gpd
from itertools import chain
from sklearn import svm

MODEL = 'GoogleNews-vectors-negative300.bin'
#you can download model from google drive if this link doesn't work (i recomend that i had a problem with downloading by terminal)
path = get_file(MODEL + '.gz', 'https://deeplearning4jblob.blob.core.windows.net/resources/wordvectors/%s.gz' % MODEL)

if not os.path.isdir('generated'):
    os.mkdir('generated')
unzipped = os.path.join('generated', MODEL)
is_model_generated=os.path.isdir('generated\GoogleNews-vectors-negative300.bin')
if is_model_generated==False:
    print("Unpacking model")
    patoolib.extract_archive(path,outdir=unzipped)


clear = lambda: os.system('cls')
clear()

print("\nLoading model to memory it will take a while\nAlso notice it will require up to 5GB of RAM\n")

model = gensim.models.KeyedVectors.load_word2vec_format(unzipped+"\GoogleNews-vectors-negative300.bin", binary=True)

countries = list(csv.DictReader(open('C:/Users/Tymon/Desktop/Python/data/countries.csv')))
countries[:10]

positive = [x['name'] for x in random.sample(countries, 40)]
negative = random.sample(model.vocab.keys(), 5000)
negative[:4]

labelled = [(p, 1) for p in positive] + [(n, 0) for n in negative]
random.shuffle(labelled)
X = np.asarray([model[w] for w, l in labelled])
y = np.asarray([l for w, l in labelled])
X.shape, y.shape


TRAINING_FRACTION = 0.7
cut_off = int(TRAINING_FRACTION * len(labelled))
clf = svm.SVC(kernel='linear')
clf.fit(X[:cut_off], y[:cut_off])

res = clf.predict(X[cut_off:])

missed = [country for (pred, truth, country) in 
 zip(res, y[cut_off:], labelled[cut_off:]) if pred != truth]

100 - 100 * float(len(missed)) / len(res), missed

all_predictions = clf.predict(model.syn0)
res = []
for word, pred in zip(model.index2word, all_predictions):
    if pred:
        res.append(word)
        if len(res) == 150:
            break
random.sample(res, 10)

country_to_idx = {country['name']: idx for idx, country in enumerate(countries)}
country_vecs = np.asarray([model[c['name']] for c in countries])
country_vecs.shape

dists = np.dot(country_vecs, country_vecs[country_to_idx['Canada']])
for idx in reversed(np.argsort(dists)[-10:]):
    print(countries[idx]['name'], dists[idx])

def rank_countries(term, topn=10, field='name'):
    if not term in model:
        return []
    vec = model[term]
    dists = np.dot(country_vecs, vec)
    return [(countries[idx][field], float(dists[idx])) 
            for idx in reversed(np.argsort(dists)[-topn:])]

print(rank_countries('cricket'))