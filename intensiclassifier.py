import nltk
from nltk.corpus import wordnet as wn, movie_reviews
import nltk.classify as cl
import re
import random
import textio

#MOVIES = [(list(movie_reviews.words(fileid)), category)
#          for category in movie_reviews.categories()
#          for fileid in movie_reviews.fileids(category)]


class Word(str):

    def __new__(cls, word_and_tag):

        items = word_and_tag.split('\\')
        return str.__new__(cls, items[0])

    def __init__(self, word_and_tag):

        items = word_and_tag.split('\\')
        self.pos = items[1]


def is_adjective(word: str):
    """Tests whether a word is (or can) be an adjective based on the classes
    provided by WordNet."""

    synsets = wn.synsets(word, pos=wn.ADJ)
    if synsets:
        return True
    else:
        return False


def is_adverb(word: str):
    """Tests whether a word is (or can) be an adverb based on the classes
    provided by WordNet."""

    synsets = wn.synsets(word, pos=wn.ADV)
    if synsets:
        return True
    else:
        return False


def frequent_adjectives(reviews: list, threshold=5, bigrams=False):
    """Extract from a list of reviews all adjectives that occur more often than
    the given threshold."""

    # fill up a FreqDist with all words from the reviews
    all_words = nltk.FreqDist(word for review in reviews for word in review[0])
    # create a set of words that occur more often than the gievn threshold
    frequent_words = {word for word, freq in all_words.items()
                      if freq >= threshold and is_adjective(word)}

    # if bigrams are requested, do it as above based on frequency
    if bigrams:
        all_bigrams = [bigram for review in reviews
                       for bigram in nltk.bigrams(review[0])]
        potentials = nltk.FreqDist(
            [(True, bigram[1]) for bigram in all_bigrams
             if (bigram[0] in {'very', 'much', 'so'} or bigram[0][-2:] == 'ly')
             and is_adjective(bigram[1])]
        )
        frequent_bigrams = {bigram for bigram, freq in potentials.items()
                            if freq >= threshold}

        frequent_words = frequent_words.union(frequent_bigrams)

    return frequent_words


def extract_features(review: list, features: set, bigrams=False):
    """Create a sparse dict of boolean values based on the given feature set for
    the given review."""

    test_list = [word for word in review]

    if bigrams:
        review_bigrams = nltk.bigrams(review)
        test_list += [(True, bigram[1]) for bigram in review_bigrams
                      if (bigram[0] in {'very', 'much', 'so'}
                          or bigram[0][-2:] == 'ly')
                      and bigram[1] in features]

    review_features = {word: (word in test_list) for word in features}

    return review_features


def extract_multinomial_features(review: list, features: set):
    """Create a sparse dict of boolean values based on the given feature set for
    the given review."""

    test_list = [word for word in review] # make a copy of the review tokens
    review_bigrams = nltk.bigrams(test_list)
    test_list += [(True, bigram[1]) for bigram in review_bigrams
                  if (bigram[0] in {'very', 'much', 'so'}
                      or bigram[0][-2:] == 'ly')
                  and bigram[1] in features]

    review_features = {word: 2 if (True, word) in test_list
                       else 1 if word in test_list
                       else 0
                       for word in features}

    return review_features


def uni_vs_bigram(data, test_proportion=0.1):

    uni_train = []
    uni_test = []
    bi_train = []
    bi_test = []

    for i in range(15):
        random.shuffle(data)
        cut = int(test_proportion * len(data))
        training, test = data[:-cut], data[-cut:]
        features = frequent_adjectives(training, threshold=20)

        # train and test a "standard" classifier
        training1 = [(extract_features(review[0], features), review[1])
                     for review in training]
        test1 = [(extract_features(review[0], features), review[1])
                 for review in test]
        classifier1 = nltk.NaiveBayesClassifier.train(training1)
        print('Unigram-classifier:')
        uni_train.append(cl.accuracy(classifier1, training1))
        print('On training:', uni_train[-1])
        uni_test.append(cl.accuracy(classifier1, test1))
        print('On test:', uni_test[-1])

        # train and test a classifier, looking at bigrams too
        training2 = [(extract_multinomial_features(review[0], features),
                      review[1])
                     for review in training]
        test2 = [(extract_multinomial_features(review[0], features),
                  review[1])
                 for review in test]
        classifier2 = nltk.NaiveBayesClassifier.train(training2)
        print('Bigram-classifier:')
        bi_train.append(cl.accuracy(classifier2, training2))
        print('On training:', bi_train[-1])
        bi_test.append(cl.accuracy(classifier2, test2))
        print('On test:', bi_test[-1])

    for number_list in [uni_train, uni_test, bi_train, bi_test]:
        print('Mean accuracy:', sum(number_list) / len(number_list))


data = textio.read_processed_data_from_file('books')
labeled_reviews = [([Word(word) for word in review],
                    'outer' if label == '1.0' or label == '5.0' else 'inner')
                   for review, label in data]
adverbs = nltk.FreqDist(
    [word1
     for review in labeled_reviews
     for word1, word2 in nltk.bigrams(review[0])
     if word1.pos in {'RB', 'RBR', 'RBS'}
     and word2.pos in {'JJ', 'JJR', 'JJS', 'VBG', 'VBN'}]
)
features = {adverb for adverb, freq in adverbs.items() if freq >= 5}
processed_reviews = [(extract_features(review, features), label)
                     for review, label in labeled_reviews]
classifier = nltk.NaiveBayesClassifier.train(processed_reviews)
print(cl.accuracy(classifier, processed_reviews))
