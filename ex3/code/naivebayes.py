import numpy as np
import math
import glob
import re
from typing import List


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
        spam_emails_count = 0
        final_dictionary = []
        priorSpam = 0
        self.logPrior = math.log(priorSpam / (1.0 - priorSpam))
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
        return ???

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
        return corr / (corr + ncorr)

    def printMostPopularSpamWords(self, num: int) -> None:
        print("{} most popular SPAM words:".format(num))
        # TODO: print the 'num' most used SPAM words from the dictionary

    def printMostPopularHamWords(self, num: int) -> None:
        print("{} most popular HAM words:".format(num))
        # TODO: print the 'num' most used HAM words from the dictionary

    def printMostindicativeSpamWords(self, num: int) -> None:
        print("{} most distinct SPAM words:".format(num))
        # TODO: print the 'num' most indicative SPAM words from the dictionary

    def printMostindicativeHamWords(self, num: int) -> None:
        print("{} most distinct HAM words:".format(num))
        # TODO: print the 'num' most indicative HAM words from the dictionary
