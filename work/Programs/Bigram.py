from nltk.corpus import webtext
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.corpus import stopwords
textwords = [w.lower () for w in webtext.words ('pirates.txt')]
finder = BigramCollocationFinder.from_words(textwords)
finder.nbest(BigramAssocMeasures.likelihood_ratio,10)
ignored_words = set (stopwords.words('english'))
filterstops = lambda w: len (w) < 3 or w in ignored_words
finder.apply_word_filter(filterstops)
finder.nbest(BigramAssocMeasures.likelihood_ratio,10)
finder.nbest(BigramAssocMeasures.likelihood_ratio,15)
