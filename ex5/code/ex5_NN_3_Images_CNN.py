import torch
from torchsummary import summary
from myNetworkTrainer import train_and_validate, writeHistoryPlots

from myImageNN import MyCNN

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device in use: ", device)

    # TODO: train and test a CNN
    print('##########################')
    print('Testing Convolutional Neural Net')
    cnnModel = MyCNN()

    criterion = None  # Cost function - torch.nn.XXX loss functions
    optimizer = None  # Optimizer algorithm - torch.optim.XXX function
    # TODO: Your might also want to change the batchSize and number of epochs depending on your optimizer configuration
    finalCNNmodel, cnnHistory = train_and_validate(cnnModel, device, criterion, optimizer, epochs=100, batchSize=512)
    writeHistoryPlots(cnnHistory, 'cnnModel', 'output/')
