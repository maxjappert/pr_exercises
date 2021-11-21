import torch
from torchsummary import summary
from myNetworkTrainer import train_and_validate, writeHistoryPlots

from myImageNN import MyFullyConnectedNN

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device in use: ", device)

    # TODO: train and test the fully connected DNN
    print('##########################')
    print('Testing Deep Neural Net')
    dnnModel = MyFullyConnectedNN()

    criterion = None  # Cost function - torch.nn.XXX loss functions
    optimizer = None  # Optimizer algorithm - torch.optim.XXX function
    # TODO: Your might also want to change the batchSize and number of epochs depending on your optimizer configuration
    finalDNNmodel, dnnHistory = train_and_validate(dnnModel, device, criterion, optimizer, epochs=100, batchSize=512)
    writeHistoryPlots(dnnHistory, 'dnnModel', 'output/')
