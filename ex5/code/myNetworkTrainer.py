import numpy as np
import time, os

import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


def writeHistoryPlots(history, modelType, filePath):
    history = np.array(history)
    plt.clf()
    plt.plot(history[:, 0:2])
    plt.legend(['Tr Loss', 'Val Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.savefig(filePath + modelType + '_loss_curve.png')
    plt.clf()
    plt.plot(history[:, 2:4])
    plt.legend(['Tr Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig(filePath + modelType + '_accuracy_curve.png')


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)



def LoadDataSet(batchSize: int) -> (int, DataLoader, int, DataLoader):
    dataset = '../data/horse_no_horse/'
    # TODO: Add data augmentation to your dataset loader for better performance
    # Remember only to add augmentation for the training set
    # If using dataset normalization, this should however also be done to the validation set

    image_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
        ]),
        'valid': transforms.Compose([
            transforms.ToTensor(),
        ])
    }


    train_directory = os.path.join(dataset, 'train')
    valid_directory = os.path.join(dataset, 'valid')

    # Number of classes
    num_classes = len(os.listdir(valid_directory)) - 1
    print(num_classes)

    # Load Data from folders
    data = {
        'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
        'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])
    }

    # Get a mapping of the indices to the class names, in order to see the output classes of the test images.
    idx_to_class = {v: k for k, v in data['train'].class_to_idx.items()}
    print(idx_to_class)

    # Size of Data, to be used for calculating Average Loss and Accuracy
    train_data_size = len(data['train'])
    valid_data_size = len(data['valid'])

    print(f"Training data size: {train_data_size}, Validation data size: {valid_data_size}, Batch size: {batchSize}")

    # Create iterators for the Data loaded using DataLoader module
    train_data_loader = DataLoader(data['train'], batch_size=batchSize, shuffle=True)
    valid_data_loader = DataLoader(data['valid'], batch_size=batchSize, shuffle=True)


    return train_data_size, train_data_loader, valid_data_size, valid_data_loader


def train_and_validate(myModel, device, criterion, optimizer, epochs: int = 25, batchSize: int = 512):
    '''
    Function to train and validate
    Parameters
        :param myModel: Model to train and validate
        :param device: Device (cpu/cuda)
        :param criterion: Loss Criterion to minimize
        :param optimizer: Optimizer for computing gradients
        :param epochs: Number of epochs (default=25)
        :param batchSize: Number of data items in a single batch (default=25)

    Returns
        model: Trained Model with best validation accuracy
        history: (dict object): Having training loss, accuracy and validation loss, accuracy
    '''
    train_data_size, train_data_loader, valid_data_size, valid_data_loader = LoadDataSet(batchSize)
    print(f"Training set size: {train_data_size}, validation set size: {valid_data_size}, Batch size: {batchSize}")
    history = []

    train_data_loader = DeviceDataLoader(train_data_loader, device)
    valid_data_loader = DeviceDataLoader(valid_data_loader, device)
    totalStartTime = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        # Set to training mode
        myModel.train()

        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0

        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs_train, labels_train) in enumerate(train_data_loader):
            y_train = labels_train[:, None]
            # Clean existing gradients
            optimizer.zero_grad()

            # Forward pass - compute outputs on input data using the model
            outputs_train = myModel(inputs_train)
            # Compute loss
            loss_train = criterion(outputs_train, y_train.float())

            # Backpropagate the gradients
            loss_train.backward()

            # Update the parameters
            optimizer.step()

            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss_train.item() * inputs_train.size(0)

            # Compute the accuracy
            pred_valid = (outputs_train > 0.5).int()

            diff = torch.abs(pred_valid.type(torch.LongTensor) - y_train.type(torch.LongTensor))
            correct_counts_train = labels_train.size(0) - diff.sum().item()

            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += correct_counts_train

        # Validation - No gradient tracking needed
        with torch.no_grad():

            # Set to evaluation mode
            myModel.eval()

            # Validation loop
            for j, (inputs_valid, labels_valid) in enumerate(valid_data_loader):
                y_valid = labels_valid[:, None]
                # Forward pass - compute outputs on input data using the model
                outputs_valid = myModel(inputs_valid)

                # Compute loss
                loss_valid = criterion(outputs_valid, y_valid.float())

                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss_valid.item() * inputs_valid.size(0)

                # Compute the accuracy
                pred_valid = (outputs_valid > 0.5).int()

                diff = torch.abs(pred_valid.type(torch.LongTensor) - y_valid.type(torch.LongTensor))
                correct_counts_valid = labels_valid.size(0) - diff.sum().item()

                # Compute total accuracy in the whole batch and add to train_acc
                valid_acc += correct_counts_valid

        # Find average training loss and training accuracy
        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        epoch_end = time.time()

        print(
            "Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start))
    totalEndTime = time.time()
    print(f"Total training time: s{totalEndTime - totalStartTime}")
    return myModel, history
