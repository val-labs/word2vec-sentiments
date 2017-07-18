from gensim import utils
from gensim.models import Doc2Vec
from TaggedLineSentence import TaggedLineSentence

# logging
import logging
import os.path
import sys
from os import environ as E
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

log = logging.getLogger()

from config import Conf

sources = {'test-neg.txt':'TEST_NEG', 'test-pos.txt':'TEST_POS', 'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS', 'train-unsup.txt':'TRAIN_UNS'}
sentences = TaggedLineSentence(sources)

model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=8)
model.build_vocab(sentences.to_array())

try:
    R = int(sys.argv[1])
except:
    R = 1   

for epoch in range(R):
    logger.info('Epoch %d' % epoch)
    model.train(sentences.sentences_perm(),
                total_examples=model.corpus_count,
                epochs=model.iter,
    )

JOIN = os.path.join

def getdir(f, d=Conf.ModelDir):
    return os.path.join(d, f)

model.save(getdir('imdb.d2v'))

import pickle

pickle.dump(sentences, open('sentences.p', 'wb'))
pickle.dump(sentences, open(getdir('sentences.p'), 'wb'))

# classifier
from sklearn.linear_model import LogisticRegression

import make_data
train_arrays, train_labels = make_data.make_train_data()
test_arrays,   test_labels = make_data.make_test_data()

log.info('Fitting')
classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)

import pickle
pickle.dump(classifier,open(getdir('cls.p'),'wb'))

print(classifier.score(test_arrays, test_labels))

classifier2 = pickle.load(open(getdir('cls.p'),'rb'))
print(classifier2.score(test_arrays, test_labels))
