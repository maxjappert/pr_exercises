import sys
from network2DHelper import Trainer
import numpy as np
import scipy.io as scio
from mySimpleNN import mySimpleNN
import torch


def parabolaData() -> None:
    X_train = np.load('../data/train_inputs.npy').T.astype(np.float32)
    y_train = np.load('../data/train_targets.npy').astype(np.float32)
    X_test = np.load('../data/validation_inputs.npy').T.astype(np.float32)
    y_test = np.load('../data/validation_targets.npy').astype(np.float32)

    nTrainSamples = X_train.shape[0]
    print("Total number of training examples: {}".format(nTrainSamples))


    # TODO: Define model, optimizer and loss
    batch_size = ???          # Define training batch size
    model = mySimpleNN()   # Initialize the 'mySimpleNN' instance
    optimizer = ???           # Define optimizer to use from torch.optim.xxx
    criterion = ???           # Define model cost function torch.nn.xxx
    epocs = ???               # Total number of epocs to execute

    trainer = Trainer(model, optimizer, criterion)
    trainer.trainModel(X_train, y_train, X_test, y_test,
                       num_of_epochs_total=epocs, batch_size=batch_size, output_folder='output/parabola/')


def flowerData() -> None:
    train = scio.loadmat('../data/flower.mat')['train']
    d, n = np.shape(train)
    y_train = np.reshape(train[-1, :], (1, n)).astype(np.float32)
    X_train = train[:-1, :].T.astype(np.float32)

    test = scio.loadmat('../data/flower.mat')['test']
    d, n = np.shape(test)
    y_test = np.reshape(test[-1, :], (1, n)).astype(np.float32)
    X_test = test[:-1, :].T.astype(np.float32)

    nTrainSamples = X_train.shape[0]
    print("Total number of training examples: {}".format(nTrainSamples))


    # TODO: Define model, optimizer and loss
    batch_size = ???          # Define training batch size
    model = mySimpleNN()   # Initialize the 'mySimpleNN' instance
    optimizer = ???           # Define optimizer to user from torch.optim.xxx
    criterion = ???           # Define model cost function torch.nn.xxx
    epocs = ???               # Total number of epocs to execute

    trainer = Trainer(model, optimizer, criterion)
    trainer.trainModel(X_train, y_train, X_test, y_test,
                       num_of_epochs_total=epocs, batch_size=batch_size, output_folder='output/flower/')


if __name__ == "__main__":
    print(sys.version)
    print("##########-##########-##########")
    print("Neural network parabola example!")
    parabolaData()
    print("##########-##########-##########")
    print("##########-##########-##########")
    print("##########-##########-##########")
    print("Neural network flower example!")
    flowerData()
    print("##########-##########-##########")
