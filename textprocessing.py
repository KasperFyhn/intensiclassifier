import nltk
import sys
import re
from typing import List, Tuple


def output_tagged_sents(tokens: List[Tuple[str, str]], out=sys.stdout, end=' '):
    """Print the tagged sentence(s), given as a list of tuples, to the out file.
    Each word will be printed in the format 'word\\TAG'."""

    words = [f'{word}\\{tag}' for word, tag in tokens]
    try:
        print(*words, file=out, end=end)
        return True
    except IOError:
        print('Could not write to stream', out)
        return False


def write_processed_data_to_file(labeled_texts: List[Tuple[list, str]], file):
    """Write POS-tagged text to a file for later retrieval. Each text in the
    list will be stripped from line shifts as each text will be separated so and
    have the format 'word1\TAG word2\TAG ... wordN\TAG #LABEL#'."""

    try:
        for text, label in labeled_texts:
            output_tagged_sents(text, out=file)
            print(f'#{label}#', file=file)
        return True
    except IOError:
        print('Could not write to stream', file)
        return False


def read_processed_data_from_file(file):
    """Read in labeled POS-tagged data from a file. Each line must be in the
    format 'word1\TAG word2\TAG ... wordN\TAG #LABEL#'."""

    with open(file) as f:
        raw = f.read()

    lines = raw.split('\n')
    labeled_texts = []
    for line in lines:
        re.findall()


sents = ['this is a test sentence which i really hope will work.',
         'but what if it doesn\'t.?', 'what will we do?']
sents = [nltk.pos_tag(nltk.word_tokenize(sent)) for sent in sents]
labeled_sents = [(sent, label) for sent, label in zip(sents, ['pos', 'neg', 'pos'])]
write_processed_data_to_file(labeled_sents, sys.stdout)
