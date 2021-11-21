import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
plt.ioff()


class Trainer():
    '''
    Neural network trainer class
    '''

    def __init__(self, model, optimizer, criterion):
        '''
        :param model:
        :param optimizer:
        :param criterion:
        '''
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def trainModel(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                   num_of_epochs_total=1000, batch_size=32, output_folder='') -> None:
        '''
        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
        :param num_of_epochs_total:
        :param batch_size:
        :param output_folder:
        :return:
        '''
        self.batch_size = batch_size

        if output_folder is not None:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
        prediction_accuracy = np.ones(num_of_epochs_total) * (-1)
        validation_loss = np.ones(num_of_epochs_total) * (-1)
        train_loss = np.ones(num_of_epochs_total) * (-1)
        # instantiate progress bar
        for epoch in range(num_of_epochs_total):
            # Training
            train_loss[epoch] = self.train_epoch(X_train, y_train)

            # Testing
            prediction_accuracy[epoch], validation_loss[epoch] = self.test_model(X_test, y_test)
            # Plot Loss and Decision Function
            if np.mod(epoch, 10) == 0:
                if output_folder is not None:
                    grid_xlim = [np.min(X_train[:, 0]), np.max(X_train[:, 0])]
                    grid_ylim = [np.min(X_train[:, 1]), np.max(X_train[:, 1])]
                    self.plot_decision_function(X_train, y_train, grid_xlim, grid_ylim,
                                                output_folder + 'tmp_' + str(epoch) + '_train')
                    self.plot_decision_function(X_test, y_test, grid_xlim, grid_ylim,
                                                output_folder + 'tmp_' + str(epoch) + '_test')
                    self.plot_loss(train_loss, validation_loss, output_folder + 'loss')
                print('Training error: %.4f Validation Accuracy epoch [%.4d/%d] %f' % (train_loss[epoch],
                                                                                       epoch, num_of_epochs_total,
                                                                                       prediction_accuracy[epoch]))

    def train_epoch(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        '''
        :param X_train:
        :param y_train:
        :return:
        '''
        self.model.train()
        train_loss = 0
        num_batches = int(X_train.shape[0] / self.batch_size)
        for batch_idx in range(0, num_batches):
            self.optimizer.zero_grad()
            # get data
            slice = self.get_ith_batch_ixs(batch_idx, X_train.shape[0], self.batch_size)
            batch_data = X_train[slice, :]
            inputs = torch.from_numpy(batch_data)
            targets = torch.t(torch.from_numpy(y_train[:, slice]))
            y_hat = self.model.forward(inputs)
            # compute loss
            loss = self.criterion(y_hat, targets)
            loss.backward()
            # make gradient update step
            self.optimizer.step()
            # keep track of training error
            train_loss += loss.item()
        total_train_loss = train_loss / num_batches
        return total_train_loss

    def test_model(self, X_evaluate: np.ndarray, y_evaluate: np.ndarray) -> (float, float):
        '''
        :param X_evaluate:
        :param y_evaluate:
        :return:
        '''
        self.model.eval()
        correct = 0
        total = 0
        samples_to_collect_cnt = 0
        num_batches = int(X_evaluate.shape[0] / self.batch_size)
        test_loss = 0

        for batch_idx in range(0, num_batches):
            slice = self.get_ith_batch_ixs(batch_idx, X_evaluate.shape[0], self.batch_size)
            batch_data = X_evaluate[slice, :]
            inputs = torch.from_numpy(batch_data)
            targets = torch.from_numpy(y_evaluate[:, slice]).T
            y_hat = self.model.forward(inputs)

            loss = self.criterion(y_hat, targets)
            test_loss += loss.item()

            pred = y_hat.clone()

            pred = (pred > 0.5).float()
            total += targets.size(0)

            diff = torch.abs(pred - targets)
            correct += targets.size(0) - diff.sum()

        acc = 100. * correct / total
        test_loss = (test_loss / num_batches)
        return acc, test_loss

    def plot_decision_function(self, X_train: np.ndarray, y_train: np.ndarray,
                               grid_xlim: list, grid_ylim: list, save_path=None) -> None:
        '''
        :param X_train:
        :param y_train:
        :param grid_xlim:
        :param grid_ylim:
        :param save_path:
        :return:
        '''
        xx, yy = np.meshgrid(np.arange(grid_xlim[0], grid_xlim[1], 0.01),
                             np.arange(grid_ylim[0], grid_ylim[1], 0.01))
        data_numpy = np.c_[xx.ravel(), yy.ravel()]
        data = torch.from_numpy(data_numpy).type(torch.FloatTensor)

        # PLOT DECISION FUNCTION
        Z = self.model(data)
        Z = (Z > 0.5).int().numpy()

        # PLOT DECISION ON TRAINING DATA
        plt.figure(figsize=(5, 5))
        plt.ylim(grid_ylim)
        plt.xlim(grid_xlim)
        tensor_x_train = torch.from_numpy(X_train).type(torch.FloatTensor)
        pred_train = self.model(tensor_x_train)
        pred_train = (pred_train > 0.5)
        X_train = X_train.T
        plt.plot(
            [
                X_train[0, i] for i in range(X_train.shape[1])
                if y_train[0, i] == 0 and pred_train[i] == 0
            ],
            [
                X_train[1, i] for i in range(X_train.shape[1])
                if y_train[0, i] == 0 and pred_train[i] == 0
            ],
            'o', color='orange', label='true negatives'
        )
        plt.plot(
            [
                X_train[0, i] for i in range(X_train.shape[1])
                if y_train[0, i] == 1 and pred_train[i] == 1
            ],
            [
                X_train[1, i] for i in range(X_train.shape[1])
                if y_train[0, i] == 1 and pred_train[i] == 1
            ],
            'o', color='red', label='true positives'
        )
        plt.plot(
            [
                X_train[0, i] for i in range(X_train.shape[1])
                if y_train[0, i] == 0 and pred_train[i] == 1
            ],
            [
                X_train[1, i] for i in range(X_train.shape[1])
                if y_train[0, i] == 0 and pred_train[i] == 1
            ],
            'o', color='blue', label='false positives'
        )
        plt.plot(
            [
                X_train[0, i] for i in range(X_train.shape[1])
                if y_train[0, i] == 1 and pred_train[i] == 0
            ],
            [
                X_train[1, i] for i in range(X_train.shape[1])
                if y_train[0, i] == 1 and pred_train[i] == 0
            ],
            'o', color='green', label='false negatives'
        )
        if np.sum(Z) > 0:
            plt.contour(xx, yy, Z.reshape(xx.shape), colors='black')
        plt.legend()
        plt.show() if save_path is None else plt.savefig(save_path + '_data')
        plt.close()

    def plot_loss(self, train_loss: np.ndarray, val_loss: np.ndarray, save_path: str) -> None:
        '''
        :param train_loss:
        :param val_loss:
        :param save_path:
        :return:
        '''
        plt.ioff()
        boo = (train_loss != -1)
        train_loss = train_loss[boo]
        val_loss = val_loss[boo]
        plt.figure(figsize=(5, 5))
        max_y = np.max(train_loss)
        if np.isnan(max_y) | np.isinf(max_y):
            print('')
        plt.ylim([0, 1])
        plt.xlim([0, train_loss.shape[0]])
        x = range(0, train_loss.shape[0])
        line1, = plt.plot(x, train_loss, label='train')
        line2, = plt.plot(x, val_loss, label='validation')
        plt.legend(handles=[line1, line2])
        plt.savefig(save_path)
        plt.close()

    def get_ith_batch_ixs(self, i: int, num_data: int, batch_size: int) -> slice:
        '''
        Split data into minibatches.
        :param i: integer - iteration index
        :param num_data: integer - number of data points
        :param batch_size: integer - number of data points in a batch
        :return: slice object
        '''
        num_minibatches = num_data / batch_size + ((num_data % batch_size) > 0)
        i = i % num_minibatches
        start = int(i * batch_size)
        stop = int(start + batch_size)
        return slice(start, stop)
