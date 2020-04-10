import re
from collections import Counter
import pandas as pd
import os.path

# initiate kata dasar from kata_dasar_kbbi
my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "./data/kata_dasar_kbbi.csv")

# create WORDS as mapping count from kata_dasar_kbbi
WORDS = Counter(list(pd.read_csv(path,header=None)[0].values))

def P(word, N=sum(WORDS.values())):
    """Probability of `word`
    
    :param word: kata
    :type word: string
    :param N: jumlah n kata, defaults to sum(WORDS.values())
    :type N: integer
    :return: Probability of word
    :rtype: float
    """

    return WORDS[word] / N

def correction(word):
    """Most probable spelling correction for word
    `flow:`
    `word -->  edits1(word) --> edits2(word) --> known(words) --> candidates(word) --> correction(word) with P as key`
    
    :param word: kata
    :type word: string
    :return: word within maximum Probability
    :rtype: string
    """
    
    return max(candidates(word), key=P)

def candidates(word):
    """Generate possible spelling corrections for word
    
    :param word: kata
    :type word: string
    :return: set of candidates words
    :rtype: set
    """
    
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words):
    """The subset of `words` that appear in the dictionary of WORDS
    
    :param words: list of word
    :type words: list
    :return: set of words that appear in the dictionary of WORDS
    :rtype: set
    """
    
    return set(w for w in words if w in WORDS)

def edits1(word):
    """All edits that are one edit away from `word`
    
    :param word: kata
    :type word: string
    :return: all kinds edit that are one edit away from `word`
    :rtype: set
    """
    
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)] # [('', 'kemarin'), ('k', 'emarin'), ('ke', 'marin'), dst]
    deletes    = [L + R[1:]               for L, R in splits if R] # ['emarin', 'kmarin', 'kearin', dst]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1] # ['ekmarin', 'kmearin', 'keamrin', dst]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters] # ['aemarin', 'bemarin', 'cemarin', dst]
    inserts    = [L + c + R               for L, R in splits for c in letters] # ['akemarin', 'bkemarin', 'ckemarin', dst]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    """All edits that are two edits away from `word`
    
    :param word: kata
    :type word: string
    :return: all kinds edit that are twice edit away from `word`
    :rtype: set
    """
    
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


