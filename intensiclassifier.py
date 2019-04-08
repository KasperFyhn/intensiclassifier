import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import movie_reviews
import nltk.classify as cl
import re
from collections import defaultdict, Counter


def main():

    intensifiers = {'very', 'really'}

    distribution = defaultdict(list)
    n_cats = defaultdict(int)
    training = processed_stars()
    #reviews = [(list(movie_reviews.words(fileid)), category)
    #           for category in movie_reviews.categories()
    #           for fileid in movie_reviews.fileids(category)]
    #training = [(Counter(nltk.bigrams(review[0])), review[1])
    #            for review in reviews]
    for doc in training:
        features = doc[0]
        n_cats[str(doc[1])] += 1
        for feature, count in features.items():
            words = feature.split('_')
            if not len(words) > 1:
                continue
            if ((words[0] in intensifiers) and
                    is_adjective(words[1])):
                distribution[str(doc[1])].append(feature * count)

    for key, dist in distribution.items():
        print(key, len(dist), n_cats[key])


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


def processed_stars(test=False,
                    categories=('books', 'dvd', 'electronics', 'kitchen')):
    """Extracts all features from the given categoires in 'processed stars' and
    return them in a list of tuple(dict(feature: count), label)."""

    if isinstance(categories, str):
        categories = [categories]

    # loop over each category and extract features and labels per line
    # append these to the final
    labeled_features = []
    for category in categories:
        # open the relevant file, either train or test
        file = f'./processed_stars/{category}/'
        if not test:
            file += 'train'
        elif test:
            file += 'test'
        with open(file, encoding='utf-8') as f:
            raw = f.read()
            # one document per line, so split into lines
            reviews = raw.split('\n')
            # extract features and their counts for each line
            features = [{ftr[0].strip(): int(ftr[1])
                         for ftr in re.findall(r'(.*?(?<!#label#)):(\d)', line)}
                        for line in reviews]
            # extract all labels
            labels = re.findall(r'#label#:(\d+.\d+)', raw)
            # zip the features list and labels into tuples and add to final list
            labeled_features += [(f_set, float(label))
                                 for f_set, label in zip(features, labels)]

    return labeled_features


main()
