import torch
from torchsummary import summary
from myNetworkTrainer import train_and_validate, writeHistoryPlots

from myImageNN import MyLogRegNN

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device in use: ", device)

    # TODO: train and test a logistic regression classifier implemented as a neural network
    print('##########################')
    print('Testing Logistic Regression')
    logRegModel = MyLogRegNN()

    criterion = None  # Cost function - torch.nn.XXX loss functions
    optimizer = None  # Optimizer algorithm - torch.optim.XXX function
    # TODO: Your might also want to change the batchSize and number of epochs depending on your optimizer configuration
    finallogRegmodel, logRegHistory = train_and_validate(logRegModel, device, criterion, optimizer, epochs=100, batchSize=512)
    writeHistoryPlots(logRegHistory, 'logRegModel', 'output/')
