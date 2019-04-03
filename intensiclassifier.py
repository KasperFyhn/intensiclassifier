import nltk.classify as cl
import re

def main():
    training = extract_processed_stars(categories='books')
    test = extract_processed_stars(test=True, categories='books')
    classifier = cl.NaiveBayesClassifier.train(training)
    print(cl.accuracy(classifier, training))
    print(cl.accuracy(classifier, test))


def extract_processed_stars(test=False,
        categories=('books', 'dvd', 'electronics', 'kitchen')):
    """Extract all features from the passed folders in 'processed stars' and
    return them in a list of tuple(dict(feature: count), label)."""

    if isinstance(categories, str):
        categories = [categories]

    labeled_features = []
    for category in categories:
        file = f'./processed_stars/{category}/'
        if not test:
            file += 'train'
        elif test:
            file += 'test'
        with open(file, encoding='utf-8') as f:
            raw = f.read()
            reviews = raw.split('\n')
            features = [{feature[0].strip(): int(feature[1])
                         for feature in re.findall(r'(.*?):(\d)', line)}
                        for line in reviews]
            labels = re.findall(r'#label#:(\d+.\d+)', raw)
            labeled_features += [(f_set, float(label))
                                 for f_set, label in zip(features, labels)]

    return labeled_features

main()
