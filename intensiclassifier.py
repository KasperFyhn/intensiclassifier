import nltk
import math
import dataio
import string
from sklearn import naive_bayes as nb, feature_selection, metrics
import pandas as pd
from collections import defaultdict
import random
import matplotlib.pyplot as plt


class Word(str):
    """A string representation in which the passed POS tag can be retrieved
    through the attribute 'pos'."""

    def __new__(cls, word_and_tag):

        items = word_and_tag.split('\\')
        return str.__new__(cls, items[0])

    def __init__(self, word_and_tag):

        items = word_and_tag.split('\\')
        self.pos = items[1]
        self.negated = False

    def set_polarity(self, value):

        self.negated = value
        return self


# private variables used by AdjDist.from_data
_prev_data = None
_prev_labeled_bigrams = None


class AdjDist(nltk.ConditionalFreqDist):

    @classmethod
    def from_data(cls, word, data_set):
        """Create an AdjDist from a labeled dataset for the given word."""

        # store adjective bigrams for next word for efficiency
        global _prev_data
        global _prev_labeled_bigrams
        if data_set is _prev_data:
            labeled_bigrams = _prev_labeled_bigrams
        else:
            labeled_bigrams = [(bigram, float(label))
                               for review, label in data_set
                               for bigram in nltk.bigrams(review)
                               if bigram[1].pos in {'JJ', 'JJR', 'JJS'}]
            _prev_data = data_set
            _prev_labeled_bigrams = labeled_bigrams

        # find all bigrams with the word as word 2
        bigrams = [(bigram, float(label)) for bigram, label in labeled_bigrams
                   if bigram[1] == word]
        cond_samples = [
                ('not' if bigram[1].negated
                 else bigram[0] if bigram[0].pos in {'RB', 'RBR', 'RBS'}
                 else 'Ø', label)
                        for bigram, label in bigrams]
        cond_samples += [('#ALL#', label) for bigram, label in bigrams]


        return cls(cond_samples)

    def copy(self, conds_higher_than=0):
        """Make a copy of the AdjDist. The kw arg conds_higher_than provides an
        option to sort out conditions with observations lower than a given
        threshold."""

        cond_samples = [(cond, obs)
                        for cond, dist in self.items()
                        for label, count in dist.items()
                        for obs in [label] * count
                        if count > conds_higher_than]
        return type(self)(cond_samples)

    def relative_frequencies(self):
        """Create a copy of the AdjDist where frequencies are given as relative
        frequencies within conditions instead of raw counts."""

        copy = self.copy()
        for cond, dist in copy.items():
            n = len(self.observations(cond))
            for key in dist:
                dist[key] = dist[key] / n

        return copy

    def clustered_distribution(self, sd_cutoff=0.1, comparison='Ø',
                               test_parameters=('mean', 'median', 'mode')):
        """Return a new AdjDist where modifiers are collapsed in clusters based
        on the cluster_conditions() method."""

        clusters = self.cluster_conditions(sd_cutoff=sd_cutoff,
                                           comparison=comparison,
                                           test_parameters=test_parameters)
        new_cond_samples = [(cluster, obs)
                            for cluster, conds in clusters.items()
                            for cond, count in conds
                            for obs in self.observations(cond)]
        new = type(self)(new_cond_samples)
        new['Ø'] = self['Ø']
        new['#ALL#'] = self['#ALL#']
        return new

    def cluster_conditions(self, sd_cutoff=0.1, comparison='Ø',
                           min_occurrence_of_mod=5,
                           test_parameters=('mean', 'median', 'mode')):
        """Return a dict with keys 'DOWN', 'AMPL' and 'UNDEC' with their values
        being sets containing the modifiers behaving as such.
        sd_cutoff is a factor multiplied with the standard deviation to give a
        a minimum value that the mean, median and mode values must diverge from
        the comparison condition."""

        # prepare the dict to be returned in the end
        clusters = {'DOWN': set(), 'AMPL': set(), 'UNDEC': set(), 'NEG': set()}

        # see if it is generally a positive or negative word
        # polarity can tell us a lot about the meaning of a word
        negations = {'not', 'never', "n't"}.intersection(
                set(self.conditions())
                )
        done = False
        for neg in negations:
            p = self.compare_conditions(neg, comparison, sd_cutoff=sd_cutoff)
            # if negation gives higher score than non-negated, word is negative
            # in which case a negative correction is used for measures onwards
            if p == 'AMPL':
                correction = -1
                done = True  # move on
                break
            # if lower score, word is positive
            elif p == 'DOWN':
                correction = 1
                done = True  # move on
                break
        # if no answer from polarity, find out if the word is pos or neg
        # based on mean, median and mode scores on the scale in general
        if not done:
            outcomes = self._possible_outcomes()
            overall_mean = sum(outcomes) / len(outcomes)
            if comparison == 'Ø' and 'Ø' not in self.conditions():
                comparison = '#ALL#'
            # if negative word, use a minus factor for measures onwards
            if self.overall_score(comparison) < overall_mean * 0.8:
                correction = -1
            elif self.overall_score(comparison) > overall_mean * 1.2:
                correction = 1
            else:
                return clusters


        # conditions of actual modifiers, i.e. excl the ALL and Ø categories
        # as well as negations which are added to its own cluster
        mod_conditions = set(self.conditions()).difference(
                {'#ALL#', 'Ø'}.union(negations)
                 )
        for neg in negations:
            clusters['NEG'].add((neg, len(self.observations(neg))))

        # run through each condition and test how it behaves compared to raw
        for cond in mod_conditions:
            if len(self.observations(cond)) > min_occurrence_of_mod:
                kind = self.compare_conditions(cond, comparison,
                                               correction=correction,
                                               sd_cutoff=sd_cutoff,
                                               test_parameters=test_parameters)
                clusters[kind].add(
                        (cond, len(self.observations(cond)))
                        )

        return clusters

    def compare_conditions(self, test_condition: str,
                           comparison_condition: str,
                           correction=1, sd_cutoff=0.1,
                           test_parameters=('mean', 'median', 'mode')):
        """Compare two conditions (i.e. modifiers) based on the given test
        parameters. If test_condition scores higher than the comparison, 'AMPL'
        is returned; if lower, 'DOWN' is returned; if unclear, 'UNDEC' is
        returned. If correction is set to -1, a lower score is higher, i.e.
        the test is an 'AMPL' if it scores lower than the comparison."""

        if test_condition not in self.conditions():
            print(test_condition, 'not present as a modifier')
            return
        if comparison_condition not in self.conditions():
            print(comparison_condition, 'not present as a modifier')
            return

        # compare over all if raw is not known
        if comparison_condition == 'Ø' and 'Ø' not in self.conditions():
            comparison_condition = '#ALL#'

        # prepare comparison and test values
        test = {'mean': self.mean(test_condition),
                'median': self.median(test_condition),
                'mode': self.mode_value(test_condition),
                'score': self.overall_score(test_condition)}
        comparison = {'mean': self.mean(comparison_condition),
                      'median': self.median(comparison_condition),
                      'mode': self.mode_value(comparison_condition),
                      'score': self.overall_score(comparison_condition)}

        results = {'DOWN': 0, 'AMPL': 0, 'UNDEC': 0}

        # test for each measure
        for measure in test_parameters:
            t = test[measure] * correction
            c = comparison[measure] * correction
            # if the difference is below the stand dev threshold
            if abs(t - c) < sd_cutoff * self.sd(comparison_condition):
                results['UNDEC'] += 1
            # if the condition is an amplifier
            elif t > c:
                results['AMPL'] += 1
            # if the condition is a downgrader
            elif t < c:
                results['DOWN'] += 1

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
            return 'UNDEC'

    def overall_score(self, condition):

        mean = self.mean(condition)
        median = self.median(condition)
        mode = self.mode_value(condition)
        return (mean + median + mode) / 3

    def mean(self, condition):
        """Return the mean value for the passed condition."""

        observations = self.observations(condition)
        if not observations:
            return
        return sum(observations) / len(observations)

    def median(self, condition):
        """Return the median for the passed condition."""

        observations = self.observations(condition)
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

        observations = self.observations(condition)
        mean = self.mean(condition)
        n = len(observations)
        if n == 1:  # if there is only one observation, return 0
            return 0
        return math.sqrt((sum((x - mean)**2 for x in observations)) / (n - 1))

    def observations(self, condition):
        """Return a sorted list of observations for the passed condition."""

        if condition not in self.conditions():
            print(condition, 'does not occur as a modifier')
            return []
        fdist = self[condition]
        return sorted([number for key, value in fdist.items()
                       for number in [key] * value])

    def _possible_outcomes(self):
        """Return the possible outcomes across all conditions."""

        return {outcome for cond in self.values() for outcome in cond.keys()}


def clusters_across_adjs(adjdists: dict, sd_cutoff=0.1, comparison='Ø',
                         min_occurrence_of_mod=5,
                         test_parameters=('mean', 'median', 'mode')):
    """Return a ConditionalFreqDist of clustering of modifiers across a range
    of AdjDists"""

    return nltk.ConditionalFreqDist(
        (mod_type, obs)
        for dist in adjdists.values()
        for mod_type, modifiers in dist.cluster_conditions(
            sd_cutoff=sd_cutoff, comparison=comparison,
            min_occurrence_of_mod=5,
            test_parameters=test_parameters
        ).items()
        for modifier, count in modifiers if not modifier == 'Ø'
        for obs in [modifier] * count
    )


def make_adj_dists(adjs, data):
    """Return a dist with adjectives as keys pointing its AdjDist based on the
    passed data."""

    n = len(adjs)
    dists = {}
    for i, adj in enumerate(adjs):
        print(f'\rMaking AdjDist {i+1} of {n}', end='')
        dists[adj] = AdjDist.from_data(adj, data)
    print()
    return dists


def load_data(file):
    """Load data from a given file where each review is stored on a line, is
    POS-tagged and finalized with #label#. The POS-tags are "hidden" from the
    strings and can be retrieved through word.pos"""

    processed_data = dataio.read_processed_data_from_file(file)
    print('Hiding POS-tags from words ...')
    return [([Word(word) for word in review], label)
            for review, label in processed_data]


def resolve_sentence_polarities(data):

    resolved_data = []

    for review, label in data:
        sentences = [[]]
        for word in review:
            if word not in string.punctuation:
                sentences[-1].append(word)
            else:
                sentences[-1].append(word)
                sentences.append([])
        resolved_sents = []
        for sentence in sentences:
            if 'not' in sentence or 'never' in sentence or "n't" in sentence:
                for word in sentence:
                    word.set_polarity(True)
            resolved_sents.append(sentence)

        flattened_sents = [word for sentence in resolved_sents
                           for word in sentence]
        resolved_data.append((flattened_sents, label))

    return resolved_data


def frequent_adjectives(data, n=None, threshold=10, bigrams=False,
                        filter_function=None):
    """Return a set of N adjectives (and adjectival verbs) that are the most
    frequent across the data and more frequent than the given threshold. If
    n_adjs is None, all adjs above the threshold will be returned."""

    # get all adjs from the data and make a frequency distribution
    adjs = nltk.FreqDist(
        filter(filter_function,
               (word for review, label in data for word in review
                if word.pos in {'JJ', 'JJR', 'JJS'})
               )
        )
    # add bigrams if requested
    if bigrams:
        adjs.update(
                filter(lambda x: True if not filter_function
                       else filter_function(x[1]),
                       (bigram for review, label in data
                        for bigram in nltk.bigrams(review)
                        if bigram[1].pos in {'JJ', 'JJR', 'JJS'}
                        and bigram[0].pos in {'RB', 'RBR', 'RBS'})
                       )
                )
    # make sure that the number of adjectives do not exceed the possible
    if n and n > len(adjs):
        n = None
    # return the n most common adjs above the threshold
    return {adj for adj, freq in adjs.most_common(n) if freq >= threshold}


def extract_features(review: list, features: set, binarized=False,
                     bigrams=False, colored=False):
    """Create a sparse vector of counts based on the given feature set for the
    given review. If 'binarized' is set, the count stops at 1 and thus also
    works as boolean. If 'bigrams' is set, bigram features are also tested.
    If 'colored' is set, the bigram features will be colored."""

    if bigrams or colored:
        # make bigrams, but keep only relevant ones - i.e. with adjectives
        relevant_bigrams = [bigram for bigram in nltk.bigrams(review)
                            if bigram[1].pos in {'JJ', 'JJR', 'JJS'}]

    if colored:
        fdist = nltk.FreqDist()
        # run over bigrams to check how/if they are modified
        for bigram in relevant_bigrams:
            mod, adj = bigram
            # look only at adjectives that are actually features
            if adj in features:
                # test for modifier only if it's an adverb
                if adj.negated:
                    kind = 'NEG'
                elif mod.pos in {'RB', 'RBR', 'RBS'} and adj in ADJ_MODIFIERS:
                    # if the modifier has been seen with the adj before
                    if mod in ADJ_MODIFIERS[adj]:
                        kind = ADJ_MODIFIERS[adj][mod]
                    # if it has been seen with other adjectives
                    elif mod in ALL_MODIFIERS:
                        kind = ALL_MODIFIERS[mod]
                    # probably something unique, set as raw
                    else:
                        kind = ''
                    # we don't want UNDEC as a feature on its own, so set as
                    # raw, but if it was UNDEC as seen with the adjective,
                    # we want to know. Therefore, the label must be kept
                    if kind == 'UNDEC':
                        kind = ''
                # if not preceded as modifier, count as raw
                else:
                    kind = ''
                # increment the count of the colored adj
                fdist[kind+adj] += 1
            else:  # if adj not in features
                continue
    elif not colored:
        fdist = nltk.FreqDist(word for word in review)

    if bigrams:
        # add the raw bigrams to the fdist
        fdist.update(bigram for bigram in relevant_bigrams)

    if binarized:
        return [1 if fdist[feature] > 0 else 0 for feature in features]
    else:
        return [fdist[feature] for feature in features]


def split_labeled_reviews(labeled_reviews):
    """Split a list of labeled reviews list(tuple(review, label)) into two
    lists and return as reviews and labels."""

    reviews = []
    labels = []
    for review, label in labeled_reviews:
        reviews.append(review)
        labels.append(label)

    return reviews, labels


def get_fold(i, data, n_folds=10):
    """Get fold i of n folds."""

    n = len(data)
    fold_size = n // n_folds
    bottom = i * fold_size
    top = bottom + fold_size
    training = data[0:bottom] + data[top:]
    test = data[bottom:top]
    return training, test


def train_bernoulliNB(training):
    """Return a trained Bernoulli Naive Bayes classifier."""

    clf = nb.BernoulliNB()
    reviews, labels = split_labeled_reviews(training)
    clf.fit(reviews, labels)
    return clf


def train_multinomialNB(training):
    """Return a trained Bernoulli Naive Bayes classifier."""

    clf = nb.MultinomialNB()
    reviews, labels = split_labeled_reviews(training)
    clf.fit(reviews, labels)
    return clf


def mutual_information(training, features):
    """Return a dict of features and their mutual information score."""

    reviews, labels = split_labeled_reviews(training)
    mutual_info = feature_selection.mutual_info_classif(reviews, labels)
    return {feature: score for feature, score in zip(features, mutual_info)}


def accuracy(clf, test):
    """Return the accuracy of the classifier on the test set."""

    reviews, labels = split_labeled_reviews(test)
    return clf.score(reviews, labels)


# load data, make folds and prepare soon-to-be dataframes
data = load_data(r"processed_data\imdb_movies")
# binarize data
# data = [(review, '0' if label in ['1.0', '2.0'] else '1')
#         for review, label in data]
random.shuffle(data)
results = defaultdict(list)  # raw accuracy
predictions = defaultdict(list)  # predictions and correct for conf matrices
n_folds = 4
n_features = 2000
measure_mutual_information = False
include_bigram_classifiers = True
filter_bigram_features = False
multi_bigrams = 4
min_n_mod_count = 5
sd_cut = 0.2
test_params = {'score'}
print_to_file = False

# prepared parameter values etc. for the different types of classifiers
# NAME, BINARIZED FEATURE VALUES, BIGRAM FEATURES, COLORED, CLASSIFIER TYPE
CLASSIFIERS = [
        ['Bernoulli', True, False, False, train_bernoulliNB],
        ['Multinomial', False, False, False, train_multinomialNB],
        ['Binarized Multinomial', True, False, False, train_multinomialNB],
        ['Colored Bernoulli', True, False, True, train_bernoulliNB],
        ['Colored Multinomial', False, False, True, train_multinomialNB],
        ['Colored B. Multinomial', True, False, True, train_multinomialNB],
        ['Bigram Bernoulli', True, True, False, train_bernoulliNB],
        ['Bigram Multinomial', False, True, False, train_multinomialNB],
        ['Bigram B. Multinomial', True, True, False, train_multinomialNB],
        ['Colored Bigram Bernoulli', True, True, True, train_bernoulliNB],
        ['Colored Bigram Multinomial', False, True, True, train_multinomialNB],
        ['Colored Bigram B. Multinom.', True, True, True, train_multinomialNB]
        ]

for i in range(n_folds):
    print('\n### RUN NUMBER', i + 1, '###')
    training, test = get_fold(i, data, n_folds=n_folds)

    # append correct test labels to predictions dict
    predictions['correct'] += [label for review, label in test]

    # prepare feature sets for later use
    if measure_mutual_information:
        print('Getting adjectives and measuring occurrence ...')
        adjectives = frequent_adjectives(training, n=n_features*2)
        # note adj occurences for each class to calculate mutual information
        adj_occurrence = [(extract_features(review, adjectives,
                                            binarized=True), label)
                          for review, label in training]

        print('Calculating mutual information to decide features ...')
        mutual_info_for_adjs = mutual_information(adj_occurrence,
                                                  list(adjectives))
        # make a feature set for this fold of the ones with highest mutual info
        fold_features = {feature
                         for feature, mut_info in sorted(
                                 mutual_info_for_adjs.items(),
                                 key=lambda x: x[1], reverse=True
                                 )[:n_features]
                         if not mut_info == 0.0
                         }
    else:
        fold_features = frequent_adjectives(training, n=n_features)
    print(f'Made unigram feature set of {len(fold_features)} of {n_features}' +
          ' possible')

    print('Making balanced dataset for AdjDists ...')
    # if one or more classes is over-represented, adj dists will be skewed
    balanced_training = dataio.make_balanced_dataset(training,
                                                     size=len(training))
    adj_dists = make_adj_dists(fold_features, balanced_training)
    # if all occurrences of a feature has been sorted out when making the
    # balanced dataset, the AdjDist will have no outcomes, so filter these
    adj_not_in_balanced = {adj for adj, d in adj_dists.items() if len(d) == 0}
    if len(adj_not_in_balanced) > 0:
        print('Some adjectives are not attested in the balanced dataset and ' +
              'will be deleted from the feature set. These are:', end=' ')
        for adj in adj_not_in_balanced:
            print(adj, end=', ')
            del adj_dists[adj]
            fold_features.remove(adj)

    print('Making modifier dicts for all features based on their clusters ...')
    # prepare a 2d dict with modifiers that are seen with the specific adjs
    # in order to retrieve their types
    ADJ_MODIFIERS = {}
    for adj, dist in adj_dists.items():
        mod_clusters = dist.cluster_conditions(
                comparison='Ø', test_parameters=test_params, sd_cutoff=sd_cut,
                min_occurrence_of_mod=min_n_mod_count
                )
        ADJ_MODIFIERS[adj] = {mod[0]: kind
                              for kind in mod_clusters
                              for mod in mod_clusters[kind]}

    # add only as colored features those mod+adj combination types that are
    # attested in the data. If not, when encountering an unseen combination,
    # the feature will be useless
    colored_features = set()
    for feature in fold_features:
        if feature in ADJ_MODIFIERS:
            for kind in set(ADJ_MODIFIERS[feature].values()):
                if kind == 'UNDEC':
                    kind = ''
                colored_features.add(kind + feature)
    colored_features = colored_features.union(fold_features)
    print(f'Made a colored feature set of {len(colored_features)} of ' +
          f'{len(fold_features) * 4} possible.')

    print('Making modifier dicts for clusters across adjectives ...')
    # prepare a dict of all seen modifiers and their types for when they have
    # not been seen before with a certain adjective
    clusters = clusters_across_adjs(adj_dists, comparison='Ø',
                                    test_parameters=test_params,
                                    sd_cutoff=sd_cut,
                                    min_occurrence_of_mod=min_n_mod_count)
    ALL_MODIFIERS = {}
    modifiers = set.union(*[set(mods.keys()) for mods in clusters.values()])
    for mod in modifiers:
        counts = {kind: dist[mod] for kind, dist in clusters.items()
                  if mod in dist.keys()}
        # if the modifier occurs very few times, across the data, it might
        # result in overfitting
        if sum(counts.values()) < min_n_mod_count:
            continue
        else:
            ALL_MODIFIERS[mod] = max(counts.keys(), key=lambda x: counts[x])

    if include_bigram_classifiers:
        if filter_bigram_features:
            # sort out unigrams that are not in the unigram features
            def bigram_filter(word):
                if word in fold_features:
                    return True
                else:
                    return False
        else:
            bigram_filter = None

        if measure_mutual_information:
            print('Getting bigrams and measuring occurrence ...')
            bigrams = frequent_adjectives(training, bigrams=True,
                                          filter_function=bigram_filter,
                                          threshold=5,
                                          n=n_features*multi_bigrams*2)
            # note bigram occurences for each class to calculate mutual info
            bigram_occurrence = [(extract_features(review, bigrams,
                                                   binarized=True), label)
                                 for review, label in training]
            print('Calculating mutual information to decide features ...')
            mutual_info_for_bigs = mutual_information(bigram_occurrence,
                                                      list(bigrams))
            # make a feature set of the bigrams with highest mutual info
            bigram_features = {feature
                               for feature, mut_info in sorted(
                                       mutual_info_for_bigs.items(),
                                       key=lambda x: x[1], reverse=True
                                       )[:n_features*multi_bigrams]
                               if not mut_info == 0.0
                               }
        else:
            bigram_features = frequent_adjectives(
                    training, bigrams=True, threshold=5,
                    n=n_features*multi_bigrams, filter_function=bigram_filter
                    )

        print(f'Made a bigram feature set of {len(bigram_features)} of ' +
              f'{n_features * multi_bigrams} possible.')

    # train and test each classifier and report results
    for classifier, binary, bigrams, colored, train_clf in CLASSIFIERS:
        print(classifier, end=': ')

        # determine the classifier's feature set depending on type
        if bigrams and colored:
            features = colored_features.union(bigram_features)
        elif bigrams:
            features = bigram_features
        elif colored:
            features = colored_features
        else:
            features = fold_features

        # process data and train
        processed_data = [(extract_features(review, features,
                                            binarized=binary, bigrams=bigrams,
                                            colored=colored), label)
                          for review, label in data]
        training, test = get_fold(i, processed_data, n_folds=n_folds)
        clf = train_clf(training)

        # test and save results
        acc = accuracy(clf, test)
        results[classifier].append(acc)
        predictions[classifier] += list(clf.predict([r for r, l in test]))
        print(acc)

if print_to_file:
    name = input('Input file name: ')
    pd.DataFrame(predictions).to_csv(
            'previous_runs/' + name + 'predictions.csv'
            )
    pd.DataFrame(results).to_csv(
            'previous_runs/' + name + 'results.csv'
            )
