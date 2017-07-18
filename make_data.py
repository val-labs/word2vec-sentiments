from gensim import utils
from gensim.models import Doc2Vec
from TaggedLineSentence import TaggedLineSentence
import numpy

from config import Conf

# classifier
from sklearn.linear_model import LogisticRegression

# logging
import logging
import sys
log = logging.getLogger()


model = None


def get_model():
    global model
    if not model:
        log.info('Model Load')
        model = Doc2Vec.load(os.path.join(Conf.ModelDir, 'imdb.d2v'))
        pass
    return model


def make_train_data():
    train_arrays = numpy.zeros((25000, 100))
    train_labels = numpy.zeros(25000)

    for i in range(12500):
        prefix_train_pos = 'TRAIN_POS_' + str(i)
        prefix_train_neg = 'TRAIN_NEG_' + str(i)
        train_arrays[i] = get_model().docvecs[prefix_train_pos]
        train_arrays[12500 + i] = get_model().docvecs[prefix_train_neg]
        train_labels[i] = 1
        train_labels[12500 + i] = 0
    return train_arrays, train_labels


def make_test_data():
    test_arrays = numpy.zeros((25000, 100))
    test_labels = numpy.zeros(25000)

    for i in range(12500):
        prefix_test_pos = 'TEST_POS_' + str(i)
        prefix_test_neg = 'TEST_NEG_' + str(i)
        test_arrays[i] = get_model().docvecs[prefix_test_pos]
        test_arrays[12500 + i] = get_model().docvecs[prefix_test_neg]
        test_labels[i] = 1
        test_labels[12500 + i] = 0

    return test_arrays, test_labels
