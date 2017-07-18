from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from random import shuffle

class TaggedLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    words = utils.to_unicode(line).split()
                    td = TaggedDocument(words, [prefix + '_%s' % item_no])
                    yield td

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    words = utils.to_unicode(line).split()
                    td = TaggedDocument(words, [prefix + '_%s' % item_no])
                    self.sentences.append(td)
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences
