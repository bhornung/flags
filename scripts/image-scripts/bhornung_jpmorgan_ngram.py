"""
Simple N-gram predictor.

It captures n words from stdin and predicts the n-th one.
The probabilities are calculated as a 2D Markov process where
state i has (n-1) words
state i+1 has n word

It is unlikely that n - 1 > 4 patterns occur to often thus
an optional parameter can limit the upper bound of the length of the first state

Tested on python 3.7
Date: 11/04/2019
Author: Dr Balazs Hornung 
"""

from collections import Counter
from itertools import chain
import sys

# hardwire train corpus

train_corpus = \
"""
Mary had a little lamb its fleece was white as snow;
And everywhere that Mary went, the lamb was sure to go. 
It followed her to school one day, which was against the rule;
It made the children laugh and play, to see a lamb at school.
And so the teacher turned it out, but still it lingered near,
And waited patiently about till Mary did appear.
"Why does the lamb love Mary so?" the eager children cry;
"Why, Mary loves the lamb, you know" the teacher did reply."
"""

# corpus cleaner

def _clean_corpus(corpus):
    """
    i) Removes all nonalphabatic characters from a corpus 
    ii) splits it to words
    Parameters:
        corpus (str) : block of text
    Returns:
        cleaned ([str]) : a list of cleaned words
    """
    
    allowed_chars = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    
    split = corpus.split()
    
    cleaned = ( "".join(filter(lambda x: x in allowed_chars, word)) for word in split)
    
    #@TODO it would be better to use a generator
    cleaned = [x.lower() for x in cleaned]
    
    return cleaned


def _generate_all_grams(corpus, max_n = 4):
    """
    Creates a generator of all n-grams
    Parameters:
        corpus ([str]) : list of words corpus
        max_n (int) : maximum length of n-gram
    Returns:
        n_gram_generator (generator) : generator of all n-grams
    """
    
    max_n_ = min(len(corpus), max_n+1)
    
    n_gram_generator = chain(*(zip(*(corpus[j:] for j in range(i))) 
                                                       for i in range(2, max_n_)))
    return n_gram_generator


def _create_n_gram_counter(n_gram_generator):
    """
    Calculates the unnormalised transition matrix as a counter.
    Parameters:
        n_gram_generator (generator) : generator of n_grams
    Returns:
        n_gram_counter (Counter) : counter of n_grams
    """
    
    # split n_grams to ((n-1), 1) pairs
    split_n_grams = ((x[:-1], x[-1]) for x in n_gram_generator)
    
    # count pairs
    # a key is (words * (n-1), word *1)
    n_gram_counter = Counter(split_n_grams)
    
    return n_gram_counter


def _create_probability_dict(n_gram_counter):
    """
    Calculates the transition probabilities as a dict of dicts.
    Parameters:
        n_gram_counter (Counter) : counter of n_grams (head, tail)
    """

    prob_dict = {}

    for (head, tail), count in n_gram_counter.items():
    
        if not head in prob_dict:
            prob_dict.update({head : {tail : count}})
        else:
            prob_dict[head].update({tail : count})

    ## normalise to unit probabilities
    for head, tail_counts in prob_dict.items():
        
        tail_counts = {tail : count * 1.0 / sum(tail_counts.values()) 
                           for tail, count in tail_counts.items()}
        
        prob_dict[head] = tail_counts
        
    return prob_dict


# @TODO add encoder before counting the words
# hashing will be much faster and comparison too.
class StaticEncoder:
    """
    Trained, unmutable encoder -- decoder pair
    """

    def __init__(self, corpus, 
                 default_missing_key = -1,
                 default_missing_val = ""):

        self._decoder = dict(enumerate(set(corpus)))

        self._encoder = {v : k for k,v in self._decoder.items()}
        
        self._default_missing_key = default_missing_key
        
        self._default_missing_val = default_missing_val
    
    
    def encode(self, x):
        """
        string to int encoding
        """
        
        return self._encoder.get(x, self._default_missing_key)
    
    
    def decode(self, x):
        """
        int to string decoding
        """
        
        return self._decoder.get(x, self._default_missing_val)

class NGramPredictor:
    """
    N-gram predictor class
    """
    
    
    def __init__(self, prob_dict):
        """
        Creates an instance of the n-gram predictor with a trained
        transition matrix.
        Parameters
            prob_dict ({str : {str : float}}) : transition probabilites
        """

        self._prob_dict = prob_dict
        
    
    def predict(self, pattern):
        """
        Lists the predictions of the subsequent word based on an input string.
        Parameters:
            pattern ((str,)) : tuple of words
        Returns:
            probabilities ({str:float}) : dictionary of probablities.
        """
        
        probabilites = self._find_probability(pattern)
        
        self._print_probabilites(probabilites)
        

    def _print_probabilites(self, probabilites):
        """
        Prints the search results.
        Parameters:
           probabilities ({str:float} or None) : probabilities of the following word.
        """
        
        if probabilites is None:
            print("No prediction available")
            
        else:
            sorted_ = [v for v in sorted(probabilites.items(), key = lambda x: (-x[1], x[0]))]
            string_components = ["{0},{1:.3f}".format(word, prob) for word, prob in sorted_]
            string = ";".join(string_components)
            print(string)
    
        
    def _find_probability(self, pattern):
        """
        Tries to find the transition probabilites.
        """
        
        result = self._prob_dict.get(pattern, None)
        
        return result
    
def _prepare_input_string(string):
    """
    Parses the input string. The input string should be in the format (n_words, word1 word2 ...)
    Parameters:
        string (str) : input string
    Returns:
        pattern ((str,)) : the meaningful text.
    """
    
    # multiple commas in string allowed
    components = string.split(',')
    
    try:
        n = int(components[0])
    except:
        raise ValueError("Cannot covert to int: malformatted string.")
        
    words = []
    for x in components[1:]:
        words.extend(x.split())
        
    pattern = tuple([x.lower() for x in words])
    
    if len(words) == 0:
        raise ValueError("No words in input string")
    
    return pattern
    
    
if __name__ == "__main__":
    
    # prepare corpus
    cleaned_corpus = _clean_corpus(train_corpus)
    corpus_encoder = StaticEncoder(cleaned_corpus)
    
    # create transition matrix
    n_gram_generator = _generate_all_grams(cleaned_corpus, max_n = 4)
    n_gram_counter = _create_n_gram_counter(n_gram_generator)
    prob_dict = _create_probability_dict(n_gram_counter)
    
    # initialise predictor
    n_gram_predictor = NGramPredictor(prob_dict)
    
    for line in sys.stdin:
        
        # clean input string
        pattern = _prepare_input_string("2,the")
        
        # predict next word
        n_gram_predictor.predict(pattern)
        