import nltk
import math
import dataio
import string
from sklearn import naive_bayes as nb, feature_selection, metrics
import pandas as pd
from collections import defaultdict
import random
import matplotlib.pyplot as plt
from adjdist import *


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


def chi_square(training, features):

    reviews, labels = split_labeled_reviews(training)
    chi_vals, p_vals = feature_selection.chi2(reviews, labels)
    return {feature: (chi, p)
            for feature, chi, p in zip(features, chi_vals, p_vals)}


def accuracy(clf, test):
    """Return the accuracy of the classifier on the test set."""

    reviews, labels = split_labeled_reviews(test)
    return clf.score(reviews, labels)


# load data, make folds and prepare soon-to-be dataframes
data = load_data(r"processed_data\books_10k")
# binarize data
# data = [(review, '0' if label in ['1.0', '2.0'] else '1')
#         for review, label in data]
random.shuffle(data)
results = defaultdict(list)  # raw accuracy
predictions = defaultdict(list)  # predictions and correct for conf matrices
n_folds = 4
n_features = 1000
measure_mutual_information = False
include_bigram_classifiers = True
filter_bigram_features = False
multi_bigrams = 4
min_n_mod_count = 2
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
