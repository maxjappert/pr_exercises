import os

import numpy as np
import math
import glob
import re
from typing import List
from pathlib import Path


class Word():
    '''
    Placeholder to store information about each entry (word) in a dictionary
    '''
    def __init__(self, word, numOfHamWords, numOfSpamWords, indicativeness):
        self.word = word
        self.numOfHamWords = numOfHamWords
        self.numOfSpamWords = numOfSpamWords
        self.indicativeness = indicativeness


class NaiveBayes():
    '''
    Naive bayes class
    Train model and classify new emails
    '''
    def _extractWords(self, filecontent: str) -> List[str]:
        '''
        Word extractor from filecontent
        :param filecontent: filecontent as a string
        :return: list of words found in the file
        '''
        txt = filecontent.split(" ")
        txtClean = [(re.sub(r'[^a-zA-Z]+', '', i).lower()) for i in txt]
        words = [i for i in txtClean if i.isalpha()]
        return words

    def train(self, msgDirectory: str, fileFormat: str = '*.txt') -> (List[Word], float):
        '''
        :param msgDirectory: Directory to email files that should be used to train the model
        :return: model dictionary and model prior
        '''
        files = sorted(glob.glob(msgDirectory + fileFormat))
        # TODO: Train the naive bayes classifier
        # TODO: Hint - store the dictionary as a list of 'wordCounter' objects
        ham_words = []
        spam_words = []
        checked_words = []
        final_dictionary = []

        # iterate over individual files
        for filename in files:
            spam = False

            # check if the message is actually spam
            if "spmsga" in filename:
                spam = True

            contents = Path(filename).read_text()

            words = self._extractWords(contents)

            # add words of each message to either the ham- or spam-bin
            for word in words:
                if spam:
                    spam_words.append(word)
                else:
                    ham_words.append(word)

        # iterate over all words and add them to the dictionary if they haven't already been added
        # while adding we iterate over both lists of words and count each occurrence
        for word in set(spam_words + ham_words):
            if not checked_words.__contains__(word):
                spam_word_counter = 0
                for candidate in spam_words:
                    if candidate == word:
                        spam_word_counter += 1
                ham_word_counter = 0
                for candidate in ham_words:
                    if candidate == word:
                        ham_word_counter += 1

                # laplacian smoothing to avoid a math domain error
                indicativeness = math.log((spam_word_counter + 1) / (ham_word_counter + 1))

                checked_words.append(word)
                final_dictionary.append(Word(word, ham_word_counter, spam_word_counter, indicativeness))

        # we use this as a prior and not the logPrior, since it seems more intuitive to simply use this for
        # the spam class and 1.0 - priorSpam for the ham class
        self.priorSpam = len(spam_words) / (len(spam_words) + len(ham_words))

        print("spam prior: ", self.priorSpam)
        self.logPrior = math.log(self.priorSpam / (1.0 - self.priorSpam))
        final_dictionary.sort(key=lambda x: x.indicativeness, reverse=True)
        self.dictionary = final_dictionary
        return self.dictionary, self.logPrior

    def classify(self, message: str, number_of_features: int) -> bool:
        '''
        :param message: Input email message as a string
        :param number_of_features: Number of features to be used from the trained dictionary
        :return: True if classified as SPAM and False if classified as HAM
        '''

        txt = np.array(self._extractWords(message))
        # TODO: Implement classification function

        spam_log_probs = []
        ham_log_probs = []

        # this is used to skip words in the message which aren't in our dictionary
        offset = 0

        for i in range(0, number_of_features):
            if not self.dictionary.__contains__(txt[i]):
                offset += 1

            # ditto
            j = i + offset

            if j >= len(txt):
                break

            # find the word in the dictionary and calculate the log probabilities for both classes from the
            # entry in the dictionary. the formulas used are copied from the slides
            for word in self.dictionary:
                if word.word == txt[j]:
                    spam_log_probs.append(math.log((word.numOfSpamWords + 1) / (word.numOfHamWords + word.numOfSpamWords + 1)))
                    ham_log_probs.append(math.log((word.numOfHamWords + 1) / (word.numOfHamWords + word.numOfSpamWords + 1)))
                    break

        # the posteriors are also calculated with formulas copied from the slides
        posterior_spam = math.log(self.priorSpam) + sum(spam_log_probs)
        posterior_ham = math.log(1.0 - self.priorSpam) + sum(ham_log_probs)

        # if the posterior for the spam class is higher, then we classify the message to said class
        return posterior_spam > posterior_ham

    def classifyAndEvaluateAllInFolder(self, msgDirectory: str, number_of_features: int,
                                       fileFormat: str = '*.txt') -> float:
        '''
        :param msgDirectory: Directory to email files that should be classified
        :param number_of_features: Number of features to be used from the trained dictionary
        :return: Classification accuracy
        '''
        files = sorted(glob.glob(msgDirectory + fileFormat))
        corr = 0  # Number of correctly classified messages
        ncorr = 0  # Number of falsely classified messages
        # TODO: Classify each email found in the given directory and figure out if they are correctly or falsely classified
        # TODO: Hint - look at the filenames to figure out the ground truth label

        for filename in files:
            contents = Path(filename).read_text()

            is_spam = self.classify(contents, number_of_features)

            # if the message is spam but wasn't classified as such or if it isn't spam and was classified as such then
            # it was falsely classified
            if ("spmsga" in filename and not is_spam) or (not "spmga" in filename and is_spam):
                ncorr += 1
            else:
                corr += 1

        assert(len(files) == corr + ncorr)

        return corr / (corr + ncorr)

    def printMostPopularSpamWords(self, num: int) -> None:
        print("{} most popular SPAM words:".format(num))
        # TODO: print the 'num' most used SPAM words from the dictionary

        temp_dictionary = self.dictionary

        temp_dictionary.sort(key=lambda x: x.numOfSpamWords, reverse=True)

        for i in range(0, num):
            print(temp_dictionary[i].word)


    def printMostPopularHamWords(self, num: int) -> None:
        print("{} most popular HAM words:".format(num))
        # TODO: print the 'num' most used HAM words from the dictionary

        temp_dictionary = self.dictionary

        temp_dictionary.sort(key=lambda x: x.numOfHamWords, reverse=True)

        for i in range (0, num):
            print(temp_dictionary[i].word)

    def printMostindicativeSpamWords(self, num: int) -> None:
        print("{} most distinct SPAM words:".format(num))
        # TODO: print the 'num' most indicative SPAM words from the dictionary

        temp_dictionary = self.dictionary

        temp_dictionary.sort(key=lambda x: x.indicativeness, reverse=True)

        for i in range (0, num):
            print(temp_dictionary[i].word)


    def printMostindicativeHamWords(self, num: int) -> None:
        print("{} most distinct HAM words:".format(num))
        # TODO: print the 'num' most indicative HAM words from the dictionary

        temp_dictionary = self.dictionary

        temp_dictionary.sort(key=lambda x: x.indicativeness, reverse=False)

        for i in range (0, num):
            print(temp_dictionary[i].word)
