import sys
import os
import matplotlib.pyplot as plt
from naivebayes import NaiveBayes


def spamHamtoyExample() -> None:
    '''
    Trains a naive bayes classifier using a folder with spam/ham emails
    Checks quality of classifier by using the model to predict the emails from the 'test' folder
    Different feature numbers are used to check how many features gives the best classification score (1-50)
    Plots the classification - x-axis = number of features, y-axis = classification accuracy
    '''
    filedir = '../data/emails/'
    naivebay = NaiveBayes()
    naivebay.train(os.path.join(filedir, 'train/'))

    numOfItemsToPrint = 10
    naivebay.printMostPopularHamWords(numOfItemsToPrint)
    naivebay.printMostPopularSpamWords(numOfItemsToPrint)
    naivebay.printMostindicativeHamWords(numOfItemsToPrint)
    naivebay.printMostindicativeSpamWords(numOfItemsToPrint)

    print('Model logPrior: {}'.format(naivebay.logPrior))
    features = [1, 2, 5, 10, 20, 30, 40, 50]
    accuracy = []
    for i in features:
        acc = naivebay.classifyAndEvaluateAllInFolder(os.path.join(filedir, 'test/'), i)
        accuracy.append(acc)
        print(i, "features, classification score:", acc)
    plt.figure("Naive results: #features vs classification error rate")
    plt.plot(features, accuracy)
    plt.grid(True)
    plt.xlabel('Number of Features')
    plt.ylabel('Classification Score')
    plt.show()


if __name__ == "__main__":
    print("Python version in use: ", sys.version)
    print("\nNaive Bayes - email classification")
    print("##########-##########-##########")
    spamHamtoyExample()
    print("##########-##########-##########")
