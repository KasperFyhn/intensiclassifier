import nltk
import sys
from typing import List, Tuple


def output_tagged_sent(tokens: List[Tuple[str, str]], out=sys.stdout):

    words = [f'{word}\\{tag}' for word, tag in tokens]
    try:
        print(*words, file=out)
        return True
    except IOError:
        print('Could not write to stream', out)
        return False



sent = 'this is a test sentence which i really hope will work.'
tokens = nltk.word_tokenize(sent)
tagged_tokens = nltk.pos_tag(tokens)
output_tagged_sent(tagged_tokens)
