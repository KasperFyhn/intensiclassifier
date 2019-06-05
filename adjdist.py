import nltk
import math
import dataio
import string
import scipy.stats as stats
import numpy


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
                        for cond in self.keys()
                        for obs in self.observations(cond)
                        if len(self.observations(cond)) > conds_higher_than]
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

    def clustered_distribution(self, threshold=0.1, comparison='Ø',
                               test_parameters=('mean', 'median', 'mode')):
        """Return a new AdjDist where modifiers are collapsed in clusters based
        on the cluster_conditions() method."""

        clusters = self.cluster_conditions(threshold=threshold,
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

    def cluster_conditions(self, threshold=0.1, comparison='Ø',
                           min_occurrence_of_mod=5,
                           test_parameters=('mean', 'median', 'mode')):
        """Return a dict with keys 'DOWN', 'AMPL' and 'UNDEC' with their values
        being sets containing the modifiers behaving as such.
        Threshold is a minimum value that the test paremeters must diverge from
        the comparison condition."""

        if comparison == 'Ø' and 'Ø' not in self.conditions():
            comparison = '#ALL#'

        # prepare the dict to be returned in the end
        clusters = {'DOWN': set(), 'AMPL': set(), 'UNDEC': set(), 'NEG': set()}

        # see if it is generally a positive or negative word
        # polarity can tell us a lot about the meaning of a word
        negations = {'not', 'never', "n't"}.intersection(
                set(self.conditions())
                )
        done = False
        for neg in negations:
            p = self.compare_conditions(neg, comparison, threshold=threshold)
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
                                               threshold=threshold,
                                               test_parameters=test_parameters)
                clusters[kind].add(
                        (cond, len(self.observations(cond)))
                        )

        return clusters

    def compare_conditions(self, test_condition: str,
                           comparison_condition: str,
                           correction=1, threshold=0.1,
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
            # if the difference is below the threshold
            if abs(t - c) < threshold:
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
        """Return an overall score of the given condition calculated as the
        mean of the mean, median and mode."""

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

    def entropy(self, condition, base=None):
        """Return the entropy of the distribution of a given condition. If base
        is set as None (which it is as default), the log base of entropy is the
        number of possible outcomes in the distribution."""

        if condition == 'Ø' and 'Ø' not in self.conditions():
            condition = '#ALL#'

        prob_dist = nltk.MLEProbDist(self[condition])
        probs = [prob_dist.prob(bin_) for bin_ in prob_dist.samples()]
        if not base:
            base = len(self._possible_outcomes())
        return stats.entropy(probs, base=base)

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


def clusters_across_adjs(adjdists: dict, threshold=0.1, comparison='Ø',
                         min_occurrence_of_mod=5,
                         test_parameters=('mean', 'median', 'mode')):
    """Return a ConditionalFreqDist of clustering of modifiers across a range
    of AdjDists"""

    return nltk.ConditionalFreqDist(
        (mod_type, obs)
        for dist in adjdists.values()
        for mod_type, modifiers in dist.cluster_conditions(
            threshold=threshold, comparison=comparison,
            min_occurrence_of_mod=min_occurrence_of_mod,
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


def resolve_sentence_polarities(data):

    resolved_data = []

    for review, label in data:
        sentences = [[]]
        for word in review:
            if word not in set.union(set(string.punctuation),
                                     {'however', 'but', 'yet'}
                                     ):
                sentences[-1].append(word)
            else:
                sentences[-1].append(word)
                sentences.append([])
        resolved_sents = []
        for sentence in sentences:
            if len(set.intersection(set(sentence),
                                    {'not', 'never', "n't", 'nothing'})
                ) > 0:
                for word in sentence:
                    word.negated = True
            resolved_sents.append(sentence)

        flattened_sents = [word for sentence in resolved_sents
                           for word in sentence]
        resolved_data.append((flattened_sents, label))

    return resolved_data

def frequent_adjectives(data, n=None, threshold=10, bigrams=False,
                        filter_function=None):
    """Return a list of N adjectives (and adjectival verbs) that are the most
    frequent across the data and more frequent than the given threshold in
    decreasing order. If n_adjs is None, all adjs above the threshold will be
    returned."""

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
    return [adj for adj, freq in adjs.most_common(n) if freq >= threshold]


def load_data(file):
    """Load data from a given file where each review is stored on a line, is
    POS-tagged and finalized with #label#. The POS-tags are "hidden" from the
    strings and can be retrieved through word.pos"""

    processed_data = dataio.read_processed_data_from_file(file)
    print('Hiding POS-tags from words ...')
    return [([Word(word) for word in review], label)
            for review, label in processed_data]
