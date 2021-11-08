import numpy as np
import math
import matplotlib.pyplot as plt
from svm import SVM


def plot_data(plt: plt, x: np.ndarray, y: np.ndarray, STYLE, label: str = ''):
    '''
    Visualize 2D data items - color according to their class
    :param plt: Plotting library to be used - ex pass plt (import matplotlib.pyplot as plt)
    :param x: 2D data
    :param y: Data labels
    :param STYLE: Marker style and color in list format, ex: [['red', '+'], ['blue', '_']]
    :param label: Obtional plot name
    '''
    unique = np.unique(y)
    for li in range(len(unique)):
        x_sub = x[:, y[0, :] == unique[li]]
        plt.scatter(x_sub[0, :], x_sub[1, :], c=STYLE[li][0], marker=STYLE[li][1], label=label + str(li))
    plt.legend()


def plot_linear_separator(plt: plt, svm: SVM, datamin: int, datamax: int):
    '''
    Visualize linear SVM separator with margins
    :param plt: Plotting library to be used - ex pass plt (import matplotlib.pyplot as plt)
    :param svm: SVM object
    :param datamin: min value on x and y axis to be shown
    :param datamax: max value on x and y axis to be shown
    '''
    x = np.arange(datamin, datamax + 1.0)
    MARG = -(svm.w[0] * x + svm.bias) / svm.w[1]
    YUP = (1 - svm.w[0] * x - svm.bias) / svm.w[1]  # Margin
    YLOW = (-1 - svm.w[0] * x - svm.bias) / svm.w[1]  # Margin
    plt.plot(x, MARG, 'k-')
    plt.plot(x, YUP, 'k--')
    plt.plot(x, YLOW, 'k--')
    for sv in svm.sv:
        plt.plot(sv[0], sv[1], 'kx')


def plot_kernel_separator(plt: plt, svm: SVM, datamin: float, datamax: float, h: float = 0.05, alpha: float = 0.25):
    '''
    :param plt: Plotting library to be used - ex pass plt (import matplotlib.pyplot as plt)
    :param svm: SVM object
    :param datamin: min value on x and y axis to be shown
    :param datamax: max value on x and y axis to be shown
    :param h: Density of classified background points
    :return:
    '''
    # function visualizes decision boundaries using color plots
    # creating meshgrid for different values of features
    xx, yy = np.meshgrid(np.arange(datamin, datamax, h), np.arange(datamin, datamax, h))
    # extracting predictions at different points in the mesh
    some = np.transpose(np.c_[xx.ravel(), yy.ravel()])
    Z = svm.classifyKernel(some)
    Z = Z.reshape(xx.shape)
    # plotting the mesh
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired, alpha=alpha, shading='auto')
    for sv in svm.sv:
        plt.plot(sv[0], sv[1], 'kx')
    plt.grid()


def get_rbg_image(image: np.ndarray) -> np.ndarray:
    img = np.zeros((32, 32, 3))
    img[:, :, 0] = np.reshape(image[:1024], (32, 32))
    img[:, :, 1] = np.reshape(image[1024:2048], (32, 32))
    img[:, :, 2] = np.reshape(image[2048:], (32, 32))

    return img


def figurePlotting(imgarray: np.ndarray, N: int, is_cifar: bool, name: str = '', random: bool = True) -> None:
    '''
    CIFAR / MNIST image visualization - rescaling the vector images to 32x32 and visualizes in a matplotlib plot
    :param imgarray: Array of images to be visualized, each column is an image
    :param N: Number of images per row/column
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


def visualizeClassification(data: np.ndarray, labels: np.ndarray, predictions: np.ndarray, num: int, is_cifar: bool,
                            name='') -> None:
    '''
    Use SVM classifier to classify images and plot a window with correctly classified and one with wrongly classified images
    :param data: CIFAR data each column is an image
    :param labels: Data labels (-1.0 or 1.0)
    :param predictions: Predicted data labels (-1.0 or 1.0)
    :param num: Number of CIFAR images to show
    :param name: Optional name of the plot
    '''
    res = np.abs(predictions - labels) / 2.0
    number_of_misses = int(np.sum(res))
    number_of_hits = int(data.shape[1] - number_of_misses)
    index = (res == 1.0).reshape(-1).astype(bool)

    missed_vectors = data[:, index]
    n_pictures = int(math.ceil(math.sqrt(min(num, number_of_misses))))

    if n_pictures > 0:
        figurePlotting(missed_vectors, n_pictures, is_cifar, name + ": Misclassified")

    index = np.invert(index)
    hit_vectors = data[:, index]
    n_pictures = int(math.ceil(math.sqrt(min(num, number_of_hits))))

    if n_pictures > 0:
        figurePlotting(hit_vectors, n_pictures, is_cifar, name + ": Correct")
    plt.show()
