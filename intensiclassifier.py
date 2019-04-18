import nltk
from nltk.corpus import movie_reviews
import math
import textio
from collections import defaultdict
import random
from pprint import pprint


class Word(str):
    """A string representation in which the passed POS tag can be retrieved
    through the attribute 'pos'."""

    def __new__(cls, word_and_tag):

        items = word_and_tag.split('\\')
        return str.__new__(cls, items[0])

    def __init__(self, word_and_tag):

        items = word_and_tag.split('\\')
        self.pos = items[1]


class AdjDist(nltk.ConditionalFreqDist):

    def __init__(self, word, data_set):
        defaultdict.__init__(self, nltk.FreqDist)

        bigrams = [(bigram, float(label))
                   for review, label in data_set
                   for bigram in nltk.bigrams(review)
                   if bigram[1] == word]
        cond_samples = [(bigram[0] if bigram[0].pos in {'RB', 'RBR', 'RBS'}
                         else 'Ø', label)
                        for bigram, label in bigrams]
        cond_samples += [('#ALL#', label) for bigram, label in bigrams]
        if cond_samples:
            for (cond, sample) in cond_samples:
                self[cond][sample] += 1

    def cluster_conditions(self, sd_cutoff=0.1, comparison='Ø',
                           test_parameters=('mean', 'median', 'mode')):
        """Return a dict with keys 'down', 'ampl' and 'undec' with their values
        being sets containing the modifiers behaving as such.
        sd_cutoff is a factor multiplied with the standard deviation to give a
        a minimum value that the mean, median and mode values must diverge from
        the comparison condition."""

        # prepare the dict to be returned in the end
        clusters = {'down': set(), 'ampl': set(), 'undec': set()}

        # see if it is generally a positive or negative word
        # polarity can tell us a lot about the meaning of a word
        done = False
        if 'not' in self.conditions():
            p = self.compare_conditions('not', comparison, sd_cutoff=sd_cutoff)
            # if negation gives higher scores than non-negated, word is negative
            if p == 'ampl':
                correction = -1
                done = True
            # if lower score, word is positive
            elif p == 'down':
                correction = 1
                done = True
        # if no answer from polarity, find out if the word is pos or neg
        # based on mean, median and mode scores on the scale in general
        if not done:
            samples = self._possible_samples()
            mean_of_samples = sum(samples) / len(samples)
            pos, neg = 0, 0
            comparison_scores = {'mean': self.mean(comparison),
                                 'median': self.median(comparison),
                                 'mode': self.mode_value(comparison)}
            for value in comparison_scores.values():
                if value >= mean_of_samples:
                    pos += 1
                elif value < mean_of_samples:
                    neg += 1
            # if negative word, add a minus factor for measures onwards
            if pos < neg:
                correction = -1
            else:
                correction = 1

        # all conditions of actual modifiers, i.e. excl the ALL and Ø categories
        mod_conditions = set(self.conditions()).difference({'#ALL#', 'Ø'})

        # run through each condition and test how it behaves compared to raw
        for cond in mod_conditions:
            kind = self.compare_conditions(cond, comparison,
                                           correction=correction,
                                           sd_cutoff=sd_cutoff,
                                           test_parameters=test_parameters)
            clusters[kind].add(cond)

        return clusters

    def compare_conditions(self, test_condition: str, comparison_condition: str,
                           correction=1, sd_cutoff=0.1,
                           test_parameters=('mean', 'median', 'mode')):
        """Compare two conditions (i.e. modifiers) based on the given test
        parameters. If test_condition scores higher than the comparison, 'ampl'
        is returned; if lower, 'down' is returned; if unclear, 'undec' is
        returned. If correction is set to -1, a lower score is higher, i.e.
        the test is an 'ampl' if it scores lower compared to the comparison."""

        if test_condition not in self.conditions():
            print(test_condition, 'not present as a modifier')
            return
        if comparison_condition not in self.conditions():
            print(comparison_condition, 'not present as a modifier')
            return

        # prepare comparison and test values
        test = {'mean': self.mean(test_condition),
                'median': self.median(test_condition),
                'mode': self.mode_value(test_condition)}
        comparison = {'mean': self.mean(comparison_condition),
                      'median': self.median(comparison_condition),
                      'mode': self.mode_value(comparison_condition)}

        results = {'down': 0, 'ampl': 0, 'undec': 0}

        # test for each measure
        for measure in test_parameters:
            t = test[measure] * correction
            c = comparison[measure] * correction
            # if the difference is below the stand dev threshold
            if abs(t - c) < sd_cutoff * self.sd(comparison_condition):
                results['undec'] += 1
            # if the condition is an amplifier
            elif t > c:
                results['ampl'] += 1
            # if the condition is a downgrader
            elif t < c:
                results['down'] += 1

        max_r = max(results.values())
        winners = []
        # find all with the max value to see if there is only one
        for key, value in results.items():
            if value == max_r:
                winners.append(key)
        if len(winners) == 1:
            return winners[0]
        # if there is not only one, it is undecided
        else:
            return 'undec'

    def mean(self, condition):
        """Return the mean value for the passed condition."""

        observations = self._observations(condition)
        if not observations:
            return
        return sum(observations) / len(observations)

    def median(self, condition):
        """Return the median for the passed condition."""

        observations = self._observations(condition)
        if not observations:
            return
        n = len(observations)
        if n == 1:
            return observations[0]
        middle = n // 2
        if n % 2 != 0:
            return observations[middle]
        else:
            return (observations[middle - 1] + observations[middle]) / 2

    def mode_value(self, condition):
        """Return the mode for the passed condition."""

        if condition not in self.conditions():
            print(condition, 'does not occur as a modifier.')
            return

        return self[condition].max()

    def sd(self, condition):
        """Return the standard deviation for the passed condition."""

        observations = self._observations(condition)
        mean = self.mean(condition)
        n = len(observations)
        if n == 1:
            print('Only one observation for', condition, '- returning 0.')
            return 0
        return math.sqrt((sum((x - mean) ** 2 for x in observations)) / (n - 1))

    def _observations(self, condition):
        """Return a sorted list of observations for the passed condition."""

        if condition not in self.conditions():
            print(condition, 'does not occur as a modifier')
            return None
        fdist = self[condition]
        return sorted([number for key, value in fdist.items()
                       for number in [key] * value])

    def _possible_samples(self):
        """Return the possible outcomes across all conditions."""

        return {sample for cond in self.values() for sample in cond.keys()}


def clusters_across_adjs(adjdists: dict, sd_cutoff=0.1, comparison='Ø',
                         test_parameters=('mean', 'median', 'mode')):
    """Return a ConditionalFreqDist of clustering of modifiers across a range of
    AdjDists"""

    return nltk.ConditionalFreqDist(
        (mod_type, modifier)
        for dist in adjdists.values()
        for mod_type, modifiers in dist.cluster_conditions(
            sd_cutoff=sd_cutoff, comparison=comparison,
            test_parameters=test_parameters
        ).items()
        for modifier in modifiers if not modifier == 'Ø'
    )


def load_data(file):
    """Load data from a given file where each review is stored on a line, is
    POS-tagged and finalized with #label#. The POS-tags are "hidden" from the
    strings and can be retrieved through word.pos"""

    return [([Word(word) for word in review], label)
            for review, label in textio.read_processed_data_from_file(file)]


def frequent_adjectives(data, n_adjs=None, threshold=5):
    """Return a set of N adjectives (and adjectival verbs) that are the most
    frequent across the data and more frequent than the given threshold. If
    n_adjs is None, all adjs above the threshold will be returned."""

    # get all adjs from the data and make a frequency distribution
    adjs = nltk.FreqDist(
        word for review, label in data for word in review
        if word.pos in {'JJ', 'JJR', 'JJS'}
    )
    # make sure that the number of adjectives do not exceed the possible
    if n_adjs and n_adjs > len(adjs):
        n_adjs = None

    return [adj for adj, freq in adjs.most_common(n_adjs) if freq >= threshold]


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


data = load_data('processed_data/books')
adjs = frequent_adjectives(data, n_adjs=100)
dists = {adj: AdjDist(adj, data) for adj in adjs}
cfd = clusters_across_adjs(dists)

"""
inf_fts = []
with open('informative_features') as f:
    raw = f.read()
    inf_fts = raw.split()

adj_dists = {adj: AdjDist(adj, data) for adj in inf_fts}
clusters = clusters_across_data(adj_dists)


classifiers = []
for i in range(10):
    print('Run number', i)
    random.shuffle(data)
    test, training = data[:1000], data[1000:]
    feature_set = frequent_adjectives(training)
    training = [(extract_features(review, feature_set), label)
                for review, label in training]
    print('\tTraining ...')
    classifier = nltk.NaiveBayesClassifier.train(training)
    print('\tTesting ...')
    training_acc = nltk.classify.accuracy(classifier, training)
    test = [(extract_features(review, feature_set), label)
            for review, label in test]
    test_acc = nltk.classify.accuracy(classifier, test)
    classifiers.append(
        {'clf': classifier, 'train_acc': training_acc,
         'test_acc': test_acc}
    )
    print(classifiers[-1])
"""
