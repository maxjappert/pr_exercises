import numpy as np
import matplotlib.pyplot as plt
from typing import List
from logreg import LOGREG


def plot2D(ax: plt, X: np.ndarray, y: np.ndarray, w: np.ndarray, name: str) -> None:
    '''
    Visualize decision boundary and data classes in 2D
    :param ax: matplotlib
    :param X: data
    :param y: data labels
    :param w: model parameters
    :param name:
    :return:
    '''
    x1 = np.array(X[1, :])  # note: X_train[0,:] is the added row of 1s (bias)
    x2 = np.array(X[2, :])
    posterior1 = LOGREG().activationFunction(w, X)
    posterior1 = np.squeeze(np.asarray(posterior1))
    markers = ['o', '+']
    groundTruthLabels = np.unique(y)
    for li in range(len(groundTruthLabels)):
        x1_sub = x1[y[:] == groundTruthLabels[li]]
        x2_sub = x2[y[:] == groundTruthLabels[li]]
        m_sub = markers[li]
        posterior1_sub = posterior1[y[:] == groundTruthLabels[li]]
        ax.scatter(x1_sub, x2_sub, c=posterior1_sub, vmin=0, vmax=1, marker=m_sub,
                   label='ground truth label = ' + str(li))
    cbar = ax.colorbar()
    cbar.set_label('posterior value')
    ax.legend()
    x = np.arange(np.min(x1), np.max(x1), 0.1)
    pms = [[0.1, 'k:'], [0.25, 'k--'], [0.5, 'r'], [0.75, 'k-.'], [0.9, 'k-']]
    for (p, m) in pms:
        yp = (- np.log((1 / p) - 1) - w[1] * x - w[0]) / w[2]
        yp = np.squeeze(np.asarray(yp))
        ax.plot(x, yp, m, label='p = ' + str(p))
        ax.legend()
    ax.xlabel('feature 1')
    ax.ylabel('feature 2')
    ax.title(name + '\n Posterior for class labeled 1')


def plot3D(ax: plt, sub3d: plt, X: np.ndarray, y: np.ndarray, w: np.ndarray, name: str) -> None:
    '''
    Visualize decision boundary and data classes in 3D
    :param ax:  matplotlib
    :param sub3d: fig.add_subplot(XXX, projection='3d')
    :param X: data
    :param y: data labels
    :param w: model parameters
    :param name: plot name identifier
    :return:
    '''
    x1 = np.array(X[1, :])  # note: X_train[0,:] is the added row of 1s (bias)
    x2 = np.array(X[2, :])
    posterior1 = LOGREG().activationFunction(w, X)
    posterior1 = np.squeeze(np.asarray(posterior1))
    markers = ['o', '+']
    groundTruthLabels = np.unique(y)
    for li in range(len(groundTruthLabels)):
        x1_sub = x1[y[:] == groundTruthLabels[li]]
        x2_sub = x2[y[:] == groundTruthLabels[li]]
        m_sub = markers[li]
        posterior1_sub = posterior1[y[:] == groundTruthLabels[li]]
        sub3d.scatter(x1_sub, x2_sub, posterior1_sub, c=posterior1_sub, vmin=0, vmax=1, marker=m_sub,
                      label='ground truth label = ' + str(li))
    ax.legend()
    x = np.arange(np.min(x1), np.max(x1), 0.1)
    pms = [[0.1, 'k:'], [0.25, 'k--'], [0.5, 'r'], [0.75, 'k-.'], [0.9, 'k-']]
    for (p, m) in pms:
        yp = (- np.log((1 / p) - 1) - w[1] * x - w[0]) / w[2]
        yp = np.squeeze(np.asarray(yp))
        z = np.ones(yp.shape) * p
        sub3d.plot(x, yp, z, m, label='p = ' + str(p))
        ax.legend()
    ax.xlabel('feature 1')
    ax.ylabel('feature 2')
    ax.title(name + '\n Posterior for class labeled 1')


def get_rbg_image(image: np.ndarray, width: int = 32) -> np.ndarray:
    img = np.zeros((width, width, 3))
    img[:, :, 0] = np.reshape(image[:1024], (width, width))
    img[:, :, 1] = np.reshape(image[1024:2048], (width, width))
    img[:, :, 2] = np.reshape(image[2048:], (width, width))
    return img


def figurePlotting(imgarray: np.ndarray, N: int, is_cifar: bool = False, name='', random=True) -> None:
    '''
    CIFAR image visualization - rescaling the vector images to 32x32 and visualizes in a matplotlib plot
    :param imgarray: Array of images to be visualized, each column is an image
    :param N: Number of images per row/column
    :param is_cifar: True if CIFAR dataset is used, False if MNIST
    :param name: Optional name of the plot
    :param random: True if the images should be taken randomly from the array - otherwise start of the array is taken
    '''
    plt.figure(name)
    for i in range(0, N * N):
        imgIndex = i
        if random:
            imgIndex = np.random.randint(low=0, high=imgarray.shape[1])
        if is_cifar:
            img = get_rbg_image(imgarray[:, imgIndex])
            plt.subplot(N, N, i + 1)
            plt.imshow(img)
            plt.axis('off')
        else:
            img = np.reshape(imgarray[:, imgIndex], (16, 16))
            plt.subplot(N, N, i + 1)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
    plt.tight_layout()


def visualizeClassification(data: np.ndarray, labels: np.ndarray, predictions: np.ndarray,
                            num: int, is_cifar: bool = False, name: str = '') -> None:
    '''
    Use LOGREG classifier to classify images and plot a window with correctly classified and one with wrongly classified images
    :param data: CIFAR data each column is an image
    :param labels: Data labels (-1.0 or 1.0)
    :param predictions: Predicted data labels (-1.0 or 1.0)
    :param num: Number of CIFAR images to show
    :param is_cifar: True if CIFAR dataset is used, False if MNIST
    :param name: Optional name of the plot
    '''
    res = np.abs(predictions - labels)
    number_of_misses = int(np.sum(res))
    number_of_hits = int(data.shape[1] - number_of_misses)
    index = (res == 1.0).reshape(-1).astype(bool)

    missed_elements = data[:, index]
    number_rows_columns = int(np.ceil(np.sqrt(min(num, number_of_misses))))

    if number_rows_columns > 0:
        figurePlotting(missed_elements, number_rows_columns, is_cifar, name + ": Misclassified")

    index = np.invert(index)
    hit_elements = data[:, index]
    number_rows_columns = int(np.ceil(np.sqrt(min(num, number_of_hits))))

    if number_rows_columns > 0:
        figurePlotting(hit_elements, number_rows_columns, is_cifar, name + ": Correct")
    plt.show()


def logreg_image(train: np.ndarray, test: np.ndarray, regularization_coefficients: List[float],
                 is_cifar: bool = False) -> None:
    '''
    without reg : 0
    with reg: regularization_coefficients = 1 / 2sigma^2
    :param train: data and labels for classifier training
    :param test: data and labels for classifier test
    :param is_cifar: True if CIFAR dataset is used, False if MNIST
    '''
    train_label = np.transpose(train[0, :].astype(np.double))
    train_label[train_label < 0] = 0.0
    train_x = train.astype(np.double)
    train_x[0, :] = 1.0

    print("Dataset ballance in training {:.2f}%".format(100 * np.sum(train_label) / len(train_label)))

    test_label = np.transpose(test[0, :].astype(np.double))
    test_label[test_label < 0] = 0.0
    test_x = test.astype(np.double)
    test_x[0, :] = 1.0

    print("Dataset ballance in test {:.2f}%".format(100 * np.sum(test_label) / len(test_label)))

    for r in regularization_coefficients:
        logreg = LOGREG(r)

        print('Training a LOGREG classifier with regularization coefficient: {}'.format(r))

        # training
        logreg.train(train_x, train_label, 50)
        print('Training')
        logreg.printClassification(train_x, train_label)
        print('Test')
        logreg.printClassification(test_x, test_label)

        visualizeClassification(train_x[1:, :], train_label, logreg.classify(train_x),
                                3 * 3, is_cifar, 'training with reg: {}'.format(r))
        visualizeClassification(test_x[1:, :], test_label, logreg.classify(test_x), 3 * 3,
                                is_cifar, 'test with reg: {}'.format(r))
