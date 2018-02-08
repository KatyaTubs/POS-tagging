
import pickle
from collections import Counter
import itertools
import numpy as np
from numpy import matlib
from scipy import sparse
import random
import time
import itertools
import scipy.misc
from enum import Enum


START_STATE = '*START*'
START_WORD = '*START*'
END_STATE = '*END*'
END_WORD = '*END*'
RARE_WORD = '*RARE_WORD*'

# Enum that holds the possible features for the MEMM model
Features = Enum('Features', 'HMM Suffixes Prefixes Hyphen Cap AllCap Num')


with open('POS_data.pickle', 'rb') as f:
    data = pickle.load(f)
with open('all_words.pickle', 'rb') as f:
    words = pickle.load(f)
with open('all_PoS.pickle', 'rb') as f:
    pos = pickle.load(f)


# append START and END to sentences
for sent in data:
    sent[0].insert(0, START_STATE)
    sent[0].append(END_STATE)
    sent[1].insert(0, START_WORD)
    sent[1].append(END_WORD)
words.append(START_WORD)
words.append(END_WORD)
pos.append(START_STATE)
pos.append(END_STATE)


N = len(data)
training_set = data[: int(0.9 * N)]
test_set = data[int(0.9 * N) + 1 :]


RARE_N = 5
sents_list = [training_set[k][1] for k in range(len(training_set))]
count_words = Counter(list(itertools.chain.from_iterable(sents_list)))

# Exchange the rare words for RARE_WORD
for sent in data:
    for i in range(len(sent[1])):
        if sent[1][i] not in count_words.keys() or count_words[sent[1][i]]\
                < RARE_N:
            sent[1][i] = RARE_WORD

words.append(RARE_WORD)

pos2i = {pos: i for (i, pos) in enumerate(pos)}
word2i = {word: i for (i, word) in enumerate(words)}
all_pos_pairs = list(itertools.product(pos, repeat = 2))



################################################################
########################## HMM #################################
################################################################

class HMM(object):
    '''
    The basic HMM_Model with multinomial transition functions.
    '''

    def __init__(self, pos_tags, words, training_set, order=1):
        '''
        The init function of the basic HMM Model.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        :param training_set: A training set of sequences of POS-tags and words.
        :param order: the order of the markov model (default is 1).
        '''

        self.words = words
        self.pos_tags = pos_tags
        self.words_size = len(words)
        self.pos_size = len(pos_tags)
        self.pos2i = {pos: i for (i, pos) in enumerate(pos_tags)}
        self.word2i = {word: i for (i, word) in enumerate(words)}
        self.order = order
        #|PoS|x|Words| sized matrix, holds emission probabilities
        self.emission = np.zeros((self.words_size, self.pos_size))
        # |PoS|x|PoS| sized matrox, holds transition probabilities
        self.transition = np.zeros((self.pos_size, self.pos_size))
        self.start_state = tuple([START_STATE] * self.order)
        self.end_state = tuple([END_STATE] * self.order)

        self.mle(training_set)

    def mle(self, training_set):
        """
        a function for calculating the Maximum Likelihood estimation of the
        transition and emission probabilities for the standard multinomial HMM.

        :param training_set: an iterable sequence of sentences, each containing
                both the words and the PoS tags of the sentence (as in the "data_example" function).
        calculates a mapping of the transition and emission probabilities for the model
        """

        # list of all pos tags, according to number of occurances
        pos_list = [training_set[k][0] for k in range(len(training_set))]
        count_pos = Counter(list(itertools.chain.from_iterable(pos_list)))

        for tagged_sent in training_set:
            for i in range(1, len(tagged_sent[0])):
                self.emission[self.word2i[tagged_sent[1][i]], [self.pos2i[tagged_sent[0][i]]]] += 1 / count_pos[
                    tagged_sent[0][i]]
                self.transition[self.pos2i[tagged_sent[0][i - 1]], self.pos2i[tagged_sent[0][i]]] += 1 / count_pos[
                    tagged_sent[0][i - 1]]

    def sample(self, n):
        '''
        Sample n sequences of words from the HMM.
        :return: A list of word sequences.
        '''

        sents = []
        for i in range(n):
            curr_pos = self.pos2i[self.start_state]
            sent = [START_WORD]
            while (self.pos_tags[curr_pos][self.order - 1] != END_STATE):
                # choose the pos tag according to transition probabilities
                new_pos = np.random.choice(a=self.pos_size, p=self.transition[curr_pos])
                # choose the word according to pos tag, with emission probabilities matrix
                word = np.random.choice(a=self.words_size, p=self.emission[:, new_pos])
                sent.append(self.words[word])
                curr_pos = new_pos
            sents.append(sent)
        return sents

    def viterbi(self, sentence):
        '''
        Given an iterable sequence of word sequences, return the most probable
        assignment of PoS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :return: iterable sequence of PoS tag sequences.
        '''
        emission = np.log(self.emission)
        transition = np.log(self.transition)

        v = transition[self.pos2i[self.start_state]] + emission[self.word2i[sentence[1]]]

        max_tags = []
        for i in range(2, len(sentence)):
            reshaped_emission = np.matlib.repmat(emission[self.word2i[sentence[i]]], self.pos_size, 1)
            reshaped_v = np.matlib.repmat(v, self.pos_size, 1).transpose()
            to_max = reshaped_emission + transition + reshaped_v
            new_v = to_max.max(axis=0)
            max_tags.append(to_max.argmax(axis=0))
            v = new_v

        tagged = []
        max_tag = self.pos2i[self.end_state]
        for i in range(len(sentence) - 3, -1, -1):
            max_tag = max_tags[i][max_tag]
            tagged.insert(0, self.pos_tags[max_tag])

        return tagged


def calc_score(tagged_set, test_set):
    """
    Calculates how many right tags in tagged set, out of all tags
    :param tagged_set: a set of viterbi tags
    :param test_set: set of sentences with right tags
    :return: how many right tags in tagged set, out of all tags
    """
    score = 0
    for i in range(len(tagged_set)):
        sent_score = 0
        tagged_sent = tagged_set[i]

        for j in range(len(tagged_sent)):
            if tagged_sent[j] == test_set[i][0][j + 1]:
                sent_score += 1

        sent_score = sent_score / float(len(tagged_sent))
        score += sent_score

    score = score / float(len(tagged_set))
    return score


def calc_non_zero(tagged_set, test_set):
    """
    Calculates the average score of tags that didn't score zero
    :param tagged_set: a set of viterbi tags
    :param test_set:  set of sentences with right tags
    :return: the average score of tags that didn't score zero
    """

    non_zero = 0
    len_non_zero = 0
    for i in range(len(tagged_set)):
        score = calc_score(tagged_set[i:i + 1], test_set[i:i + 1])
        if score != 0:
            non_zero += score
            len_non_zero += 1

    return non_zero / len_non_zero


# Pre-process the data before initializing bigram HMM objects
bigram_data = []
for sent in data:
    tags = [tuple([tag]) for tag in sent[0]]
    bigram_data.append([tags, sent[1]])

bigram_training_set = bigram_data[: int(0.9 * N)]
bigram_test_set = bigram_data[int(0.9 * N) + 1 :]
bigram_pos_tags = []
for pos_tag in pos:
    bigram_pos_tags.append(tuple([pos_tag]))
bigram_hmm = HMM(bigram_pos_tags, words, bigram_training_set, order=1)
tags = []
for i in range(len(bigram_test_set[0:1])):
    tags.append(bigram_hmm.viterbi(bigram_test_set[i][1]))
print(calc_score(tags, bigram_test_set[0:1]))


# Pre-process the data before initializing trigram HMM objects
temp_data = data
for sent in temp_data:
    sent[0].insert(0, START_STATE)
    sent[1].append(END_WORD)
    sent[0].append(END_STATE)
    tags = [tuple([sent[0][k - 1], sent[0][k]]) for k in range(1, len(sent[0]))]
    trigram_data.append([tags, sent[1]])

trigram_training_set = trigram_data[: int(0.9 * N)]
trigram_test_set = trigram_data[int(0.9 * N) + 1:]
trigram_pos_tags = list(itertools.product(pos, repeat = 2))
trigram_hmm = HMM(trigram_pos_tags, words, trigram_training_set, order=2)
tags = []
for i in range(len(trigram_test_set[0:1])):
    tags.append(trigram_hmm.viterbi(trigram_test_set[i][1]))
print(calc_score(tags, trigram_test_set[0:1]))


################################################################
########################## MEMM ################################
################################################################

class SparseVector:
    """
    A class that represents a sparse vector.
    Holds the size of the vector, a numpy array with values different than 0,
    and a dictionary from the real locations of values different than 0 to 
    the locations in the numpy array.
    """
    def __init__(self, dim, values_per_idx = {}):
        # holds non-zero values of the vector
        self.values_array = np.zeros(len(list(values_per_idx.keys())))
        # holds dictionary from indexes of non-zero values in the vector to
        # their places in the numpy array
        self.indexes = {}
        # the dimension of the vector
        self.dim = dim
        i=0
        for idx in values_per_idx.keys():
            self.indexes[idx] = i
            self.values_array[i] = values_per_idx[idx]
            i += 1


    def plus(self, other):
        """
        Adds other to self, in place
        """
        if self.dim != other.dim:
            return None
        i = self.values_array.size
        diff = [k for k in other.indexes.keys() if k not in self.indexes.keys()]
        self.values_array.resize(i+len(diff))
        for key in diff:
            self.indexes[key] = i
            i += 1
        self.values_array[[self.indexes[k] for k in list(other.indexes.keys())]]\
            += other.values_array[list(other.indexes.values())]
        return self

    
    def minus(self, other):
        """
        Substructs other from self, in place
        """
        if self.dim != other.dim:
            return None
        i = self.values_array.size
        diff = [k for k in other.indexes.keys() if k not in self.indexes.keys()]
        self.values_array.resize(i+len(diff))
        for key in diff:
            self.indexes[key] = i
            i += 1
        self.values_array[[self.indexes[k] for k in list(other.indexes.keys())]]\
            -= other.values_array[list(other.indexes.values())]
        return self


    def sum_all(self):
        """
        Sums all the values in the vector
        """
        return np.sum(self.values_array)

    def divide(self, n):
        """
        Divides the values in the vector by the given value
        """
        self.values_array = self.values_array / float(n)
        return self

    def multiply(self, n):
        """
        Multiplies the values in the vector by a given value
        """
        self.values_array = self.values_array * n
        return self
    
    def get(self, keys):
        """
        :param keys: a list of indexes to return
        :return: a list of values in the vector according to given indexes
        """
        return [self.values_array[self.indexes[k]] if k in self.indexes.keys()
                else 0 for k in keys]



class MEMM:
    '''
    The base Maximum Entropy Markov Model with log-linear transition functions.
    '''

    def __init__(self, pos_tags, words, bool_features):
        '''
        The init function of the MEMM.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        :param bool_features: a boolean array, that for each Enum feature holds whether we want to include
                              it in our features or not.
        '''

        self.words = words
        self.pos = pos_tags
        self.words_size = len(words)
        self.pos_size = len(pos)
        self.pos2i = {pos: i for (i, pos) in enumerate(pos)}
        self.word2i = {word: i for (i, word) in enumerate(words)}
        self.bool_features = bool_features
        # list of all possible pairs of the pos tags
        self.pos_pairs = list(itertools.product(self.pos, repeat = 2))
        self.dim_hmm = len(pos)*len(pos)+len(pos)*len(words)
        # the lenght of the feature vector
        self.dim = self.dim_hmm + bool_features[Features.Suffixes]*len(SUFFIXES) \
                   + bool_features[Features.Prefixes]*len(PREFIXES) +\
                   bool_features[Features.Hyphen]*1 +\
                   bool_features[Features.Cap]+bool_features[Features.AllCap]+\
                   bool_features[Features.Num]

    def get_hmm_indexes(self, tag_pair, word):
        """
        Returns the indexes in the feature vector that hold the values for (prev_tag, this_tag)
        and (this_tag, word)
        :param tag_pair: (prev_tag, this_tag
        :param word:
        :return:
        """
        return [(pos2i[tag_pair[0]]*len(self.pos))+pos2i[tag_pair[1]],
                len(self.pos)*2 + (pos2i[tag_pair[1]]*len(self.words)) +
                self.word2i[word]]

    def get_hmm_table(self, word, weight):
        """
        returns a table with the sum of transition and emission probabilities in each entry -
        each row corresponds to a previous tag, and each column corresponds to the current possible tag,
        so an entry [i,j] in the matrix is sum of transition(tag_i, tag_j) + emission(tag_j, word)
        :param word: the current word for which we calculate the probabilities
        :param weight: the weight of the feature vector
        :return:
        """
        table =  np.array([sum(weight.get([(pos2i[tag_pair[0]]*len(
            self.pos))+pos2i[tag_pair[1]], len(self.pos)*2 + (pos2i[tag_pair[1]]
                                *len(self.words)) + self.word2i[word]]))
                                 for tag_pair in self.pos_pairs])
        return np.reshape(table, (self.pos_size, self.pos_size))

    def get_feature_indexes(self, word):
        """
        Returns indexes of appropriate entries in the feature vector, according to the given word.
        :param word: a word for which we calculate the features
        :return: indexes of appropriate entries in the feature vector
        """
        inds = []
        if self.bool_features[Features.Suffixes]:
            for suf in SUFFIXES:
                if word.endswith(suf):
                    inds.append(self.dim_hmm + i)
        if self.bool_features[Features.Prefixes]:
            for pref in PREFIXES:
                if word.startswith(pref):
                    inds.append(self.dim_hmm +
                        self.bool_features[Features.Suffixes]*len(SUFFIXES) +i)
        if self.bool_features[Features.Hyphen]:
            if '-' in word:
                inds.append(self.dim_hmm + self.bool_features[Features.Suffixes]
                        *len(SUFFIXES) + self.bool_features[Features.Prefixes]*
                         len(PREFIXES))
        if self.bool_features[Features.Cap]:
            if word[0].isupper():
                inds.append(self.dim_hmm +self.bool_features[Features.Suffixes]*
                        len(SUFFIXES) + self.bool_features[Features.Prefixes]*
                            len(PREFIXES) + self.bool_features[Features.Hyphen])
        if self.bool_features[Features.AllCap]:
            if word.isupper():
                inds.append(self.dim_hmm + self.bool_features[Features.Suffixes]
                *len(SUFFIXES) + self.bool_features[Features.Prefixes]*len(
                    PREFIXES) + self.bool_features[Features.Hyphen] +
                            self.bool_features[Features.Cap])
        if self.bool_features[Features.Num]:
            if any(char.isdigit() for char in word):
                inds.append(self.dim_hmm + self.bool_features[Features.Suffixes]*
                        len(SUFFIXES) + self.bool_features[Features.Prefixes]*
                            len(PREFIXES) + self.bool_features[Features.Hyphen]
                            + self.bool_features[Features.Cap] +
                            self.bool_features[Features.AllCap])
        return inds

    def get_features_table(self, word, weight):
        """
        Returns the sum of the transition-emission table, and sum of all other differnt features
        Essentially returns a matrix of dot products of the feature vector and the weight vector
        :param word: the word for which we want to calculate the probabilities table
        :param weight: the weight with wich we take the dot product of the feature vectors
        :return: table of dot products of feature vector and weight
        """
        return self.get_hmm_table(word, weight) + \
               sum(weight.get(self.get_feature_indexes(word)))

    def feature_function(self, this_tag, prev_tag,  word):
        """
        Returns a SparseVector object that represents the feature vector with 1 values
        in places that correspond to (prev_tag, this_tag_, (this_tag, word) and different features of the word.
        :param this_tag:
        :param prev_tag:
        :param word:
        :return:
        """
        values = {}
        inds = self.get_hmm_indexes((prev_tag, this_tag), word) + \
               self.get_feature_indexes(word)
        for i in inds:
            values[i] = 1
        return SparseVector(self.dim, values)


# list of possible suffixes
SUFFIXES = ['acy', 'al', 'ence', 'ance', 'dom', 'er', 'or', 'ism', 'ist', 'ty',
            'ment', 'ness', 'ship', 'sion', 'tion', 'ate', 'en', 'fy', 'ise',
            'ize', 'ble', 'esque', 'ful', 'ic', 'ical', 'ous', 'ish', 'ive',
            'less', 'y']
# list of possible prefixes
PREFIXES = ['anti', 'auto', 'de', 'dis', 'down', 'extra', 'hyper', 'il', 'im',
            'in', 'ir', 'inter', 'mega', 'mid', 'mis', 'non', 'over', 'out',
            'post', 'pre', 'pro', 're', 'semi', 'sub', 'super', 'tele', 'trans',
            'ultra', 'un', 'under', 'up']


def memm_viterbi(sentence, weight, model):
    """
    Implements the viterbi algorithm for MEMM model
    :param sentence: the sentence to tag
    :param weight: weight according to which to calculate the tags
    :param model: the model that holds the essential parameters for tagging
    :return: calculated tags for the sentence
    """

    # Calculate the table for all possible states, but take only the row corresponding
    # to START_STATE as previous tag.
    v = model.get_features_table(sentence[1], weight)[model.pos2i[START_STATE]]
    v = v - scipy.misc.logsumexp(v) # normalization

    # list in which we will store the maximal previous tags
    max_tags = []
    for i in range(2, len(sentence)):
        all_tags = model.get_features_table(sentence[i], weight)
        all_tags =  all_tags - np.matlib.repmat(scipy.misc.logsumexp(all_tags,
                                                axis = 1), model.pos_size, 1)
        to_max = np.matlib.repmat(v, model.pos_size, 1).transpose() + all_tags
        max_tags_index = np.argmax(to_max, axis = 0)
        v = np.max(to_max, axis = 0)
        max_tags.append(max_tags_index)

    tagged = []
    max_tag = model.pos2i[END_STATE]
    for i in range(len(sentence) - 3, -1, -1):
        max_tag = max_tags[i][max_tag]
        tagged.insert(0, model.pos[max_tag])
    tagged.append(END_STATE)
    tagged.insert(0, START_STATE)

    return tagged


ITERATIONS = 4

def perceptron(training_set, model, learning_rate = 1):
    """
    Implementation of the perceptron algorithm
    :param training_set:
    :param model:
    :return: the weights for tagging the data
    """

    dim = model.dim
    weight = SparseVector(dim)
    sum_weights = SparseVector(dim)
    
    for n in range(ITERATIONS):
        print('n now is:', n)
        inds = list(range(len(training_set)))
        random.shuffle(inds)
        for j in range(len(inds)):

            if (j % int((len(inds)/100))) == 0:
                print(int(j/len(inds)*100), "done")
            i = inds[j]
            sent = training_set[i][1]
            sent_true_tags = training_set[i][0]
            sent_viterbi_tags = memm_viterbi(sent, weight, model)
            viterbi_feature = SparseVector(dim)
            gold_feature = SparseVector(dim) # the optimal feature

            for k in range(1, len(sent_viterbi_tags)):
                viterbi_feature = viterbi_feature.plus(
                    model.feature_function(this_tag =  sent_viterbi_tags[k],
                                           prev_tag = sent_viterbi_tags[k-1],
                                           word = sent[k]))
                gold_feature = gold_feature.plus(model.feature_function(
                    this_tag = sent_true_tags[k], prev_tag =
                    sent_true_tags[k-1], word = sent[k]))

            feature_diff = gold_feature.minus(viterbi_feature)
            feature_diff = feature_diff.multiply(learning_rate)
            weight = weight.plus(feature_diff)

            # instead of adding the weight, we can calculate formula for adding only
            # the difference between this weight and previos weight, which is much lighter
            sum_weights = sum_weights.plus(feature_diff.multiply((len(
                training_set)-j) + (len(training_set)*(ITERATIONS-n-1))))

    sum_weights = sum_weights.divide(len(training_set) * ITERATIONS)
    return sum_weights



def evaluate(model, weights, test_set):
    """
    Function that evaluates the score of the learned weights
    :param model: model according to which we tag
    :param weights: the weigts to evaluate
    :param test_set:
    :return: number of the right tags, out of all pags
    """
    total_score = 0
    total_tags = 0
    for j in range(len(test_set)):
        sent = test_set[j][1]
        tagged_sent = test_set[j][0]
        tagged_viterbi = memm_viterbi(sent, weights, model)
        for k in range(0, len(tagged_viterbi)):
            if tagged_sent[k] == tagged_viterbi[k]:
                total_score += 1
        total_tags += len(tagged_viterbi)

    return total_score / total_tags


# List of dictionaries with different features, for initializing different MEMM models
hmm_dict = {Features.HMM: True, Features.Suffixes: False,
            Features.Prefixes: False, Features.Hyphen : False, Features.Cap
            : False, Features.AllCap: False, Features.Num:False}
full_dict = {Features.HMM: True, Features.Suffixes: True, Features.Prefixes:
            True, Features.Hyphen : True, Features.Cap : True, Features.AllCap:
            True,  Features.Num:True}
suffixes_dict = {Features.HMM: True, Features.Suffixes: True,
                 Features.Prefixes: False, Features.Hyphen : False,
                 Features.Cap : False, Features.AllCap: False,
                 Features.Num:False}
prefixes_dict = {Features.HMM: True, Features.Suffixes: False,
                 Features.Prefixes: True, Features.Hyphen : False,
                 Features.Cap : False, Features.AllCap: False,
                 Features.Num:False}
hyphen_dict = {Features.HMM: True, Features.Suffixes: False,
               Features.Prefixes: False, Features.Hyphen : True,
               Features.Cap : False, Features.AllCap: False, Features.Num:False}
cap_dict = {Features.HMM: True, Features.Suffixes: False,
            Features.Prefixes: False, Features.Hyphen : False,  Features.Cap
            : True, Features.AllCap: False, Features.Num:False}
all_cap_dict = {Features.HMM: True, Features.Suffixes: False,
                Features.Prefixes: False, Features.Hyphen : False,
                Features.Cap : False, Features.AllCap: True, Features.Num:False}
num_dict = {Features.HMM: True, Features.Suffixes: False,
            Features.Prefixes: False, Features.Hyphen : False, Features.Cap
            : False, Features.AllCap: False, Features.Num:True}

import tensorflow

