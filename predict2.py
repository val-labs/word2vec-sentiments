from gensim import utils
from gensim.models import Doc2Vec
from TaggedLineSentence import TaggedLineSentence
import numpy
import pickle

# classifier
from sklearn.linear_model import LogisticRegression

# logging
import logging
import sys
log = logging.getLogger()
log.setLevel(logging.INFO)

log.info('Model Load')
model = Doc2Vec.load('./imdb.d2v')

log.info('Sentiment')

reviews = [
    "bad terrible awful",
    "great classic interesting",
    "boring long stupid predictable",
    "new fun wonderful innovating",
    ]

aggregated = numpy.zeros(len(reviews))
print(aggregated)
total_runs = 0

import make_data
test_arrays,   test_labels = make_data.make_test_data()

log.info('Predicting')

classifier2 = pickle.load(open('cls.p','rb'))

print(classifier2.score(test_arrays, test_labels))

def doloop():
    global total_runs
    total_runs += 1
    print('===')
    for n,r in enumerate(reviews):
        v = model.infer_vector(r.split())
        p = classifier2.predict([v])
        print("RA", p, r)
        aggregated[n] += p

for n in range(1000):
    doloop()

print(total_runs, aggregated/total_runs)
