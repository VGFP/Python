import os
from keras.utils import get_file
import gensim
import patoolib
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
figsize(10, 10)

from sklearn.manifold import TSNE
import json
from collections import Counter
from itertools import chain


MODEL = 'GoogleNews-vectors-negative300.bin'

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

#search_similarities=input("Write the word and the model will show similar words: ")

#print("\nSearching for similar words it may take a while\n")

#search_res=model.most_similar(positive=[search_similarities])

#print(search_res)

def A_is_to_B_as_C_is_to(a, b, c, topn=1):
  a, b, c = map(lambda x:x if type(x) == list else [x], (a, b, c))
  res = model.most_similar(positive=b + c, negative=a, topn=topn)
  if len(res):
    if topn == 1:
      return res[0][0]
    return [x[0] for x in res]
  return None

A_is_to_B_as_C_is_to('man', 'woman', 'king')

for country in 'Italy', 'France', 'India', 'China':
    print("\n")
    print('%s is the capital of %s' %
          (A_is_to_B_as_C_is_to('Germany', 'Berlin', country), country))

for company in 'Google', 'IBM', 'Boeing', 'Microsoft', 'Samsung':
  products = A_is_to_B_as_C_is_to(
    ['Starbucks', 'Apple'], ['Starbucks_coffee', 'iPhone'], company, topn=3)
  print("\n")
  print('%s -> %s' %
        (company, ', '.join(products)))

beverages = ['espresso', 'beer', 'vodka', 'wine', 'cola', 'tea']
countries = ['Italy', 'Germany', 'Russia', 'France', 'USA', 'India']
sports = ['soccer', 'handball', 'hockey', 'cycling', 'basketball', 'cricket']

items = beverages + countries + sports

item_vectors = [(item, model[item])
                    for item in items
                    if item in model]
vectors = np.asarray([x[1] for x in item_vectors])
lengths = np.linalg.norm(vectors, axis=1)
norm_vectors = (vectors.T / lengths).T
tsne = TSNE(n_components=2, perplexity=10,
            verbose=2).fit_transform(norm_vectors)
x=tsne[:,0]
y=tsne[:,1]

fig, ax = plt.subplots()
ax.scatter(x, y)

for item, x1, y1 in zip(item_vectors, x, y):
    ax.annotate(item[0], (x1, y1))


plt.show()

input("Press enter to end")