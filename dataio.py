import nltk
import sys
import re
from typing import List, Tuple


def output_tagged_sents(tokens: List[Tuple[str, str]], out=sys.stdout,
                        end=' '):
    """Print the tagged sentences, given as a list of tuples, to the out file.
    Each word will be printed in the format 'word\\TAG'."""

    # make a list of correctly formatted string
    words = [f'{word}\\{tag}' for word, tag in tokens]
    try:
        print(*words, file=out, end=end)
        return True
    except IOError:
        print('Could not write to stream', out)
        return False


def write_processed_data_to_file(labeled_texts: List[Tuple[list, str]], file):
    """Write POS-tagged text to a file for later retrieval. Each text in the
    list will be stripped from line shifts as each text will be separated so
    and have the format 'word1\TAG word2\TAG ... wordN\TAG #LABEL#'."""

    try:
        for text, label in labeled_texts:
            output_tagged_sents(text, out=file)
            print(f'#{label}#', file=file)
        return True
    except IOError:
        print('Could not write to stream', file)
        return False


def extract_amazon_reviews(review_file):
    """Extract reviews from a structured XML file returned as
    list(review, tag)."""

    print(f'Decoding {review_file} ...')
    with open(review_file, encoding='ISO-8859-1') as f:
        raw = f.read()

    print('Extracting review text and labels ...')
    # extract each review and parse them
    reviews = re.findall(r'<review>(.+?)</review>', raw, flags=re.DOTALL)
    reviews = [re.findall(
        r'<rating>(.+?)</rating>.*?<review_text>(.+?)</review_text>', review,
        flags=re.DOTALL
    )[0] for review in reviews]
    # get cleaned review text and the tag
    reviews = [(re.sub(r'\s+', ' ', review[1]).lower().strip(),
                review[0].strip()) for review in reviews]

    return reviews


def extract_imdb_reviews(review_file):
    """Extract reviews from a structured file returned as list(review, tag)."""

    print(f'Decoding {review_file} ...')
    with open(review_file, encoding='utf-8') as f:
        raw = f.read()

    print('Extracting review text and labels ...')
    trash = {'<sssss>', '-rrb-', '-lrb-'}
    lines = raw.split('\n')[:-1]
    reviews = []
    for line in lines:
        chunks = line.split('\t\t')
        label = chunks[2]
        review = ' '.join(w for w in chunks[3].split() if w not in trash)
        reviews.append((review, label))

    return reviews


def process_reviews(reviews: list):
    """Return a list of the passed reviews with each review being tokenized and
    POS-tagged."""

    def process_review(review, i, n):
        print(f'\rProcessing {i + 1} of {n} reviews', end='')
        return nltk.pos_tag(nltk.word_tokenize(review[0].strip())), review[1]

    n = len(reviews)
    processed = [process_review(review, i, n)
                 for i, review in enumerate(reviews)]
    return processed


def make_balanced_dataset(data: list, size=10000):
    """Return a processed and balanced from the passed list of data"""

    # find out how many categories in the data and determine category size
    categories = {label: 0 for review, label in data}
    for review, label in data:
        categories[label] += 1

    # determine category size. If balance not possible, set to highest possible
    cat_size = size / len(categories)
    min_val = min(categories.values())
    if cat_size > min_val:
        cat_size = min_val
        size = min_val * len(categories)
        print('There are categories with too few reviews to make an even ' +
              'dataset of the requested size. Each category will contain ' +
              f'{min_val} reviews, giving a total of {size} reviews.')

    # reset category counter for the iteration
    categories = {cat: 0 for cat in categories}
    # iterate over the data and pick reviews until there are enough of all cats
    balanced_list = []
    reviews = iter(data)
    while len(balanced_list) < size:
        review = next(reviews)
        cat = review[1]
        if not categories[cat] >= cat_size:
            categories[cat] += 1
            balanced_list.append(review)

    return process_reviews(balanced_list)


def read_processed_data_from_file(file, encoding='latin1'):
    """Read in labeled POS-tagged data from a file. Each line must be in the
    format 'word1\TAG word2\TAG ... wordN\TAG #LABEL#'."""

    with open(file, encoding=encoding) as f:
        raw = f.read()

    lines = raw.split('\n')
    labeled_texts = []
    n = len(lines) - 1
    for i, line in enumerate(lines):
        print(f'\rLoading review {i} of {n}', end='')
        if line == '':
            continue
        tagged_words = re.findall(r'(.+?\\.+?) ', line)
        label = re.findall(r'#(\d+.\d)#', line)[0]
        labeled_texts.append((tagged_words, label))
    print()
    return labeled_texts


def read_run_info_from_file(file):
    """Return a list of dicts with training and test accuracy and 2000 most
    informative features from an earlier run."""

    with open(file) as f:
        raw = f.read()

    lines = raw.split('\n')
    runs = []
    for line in lines:
        if line == '':
            continue
        comma_splits = line.split(',')
        train_acc = float(comma_splits[0].strip())
        test_acc = float(comma_splits[1].strip())
        feature_set = set(w for w in comma_splits[2].split() if not w == '')
        runs.append(
            {'fts': feature_set, 'train_acc': train_acc, 'test_acc': test_acc}
        )
    return runs


def processed_stars(test=False,
                    categories=('books', 'dvd', 'electronics', 'kitchen')):
    """Extracts all features from the given categories in 'processed stars' and
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
