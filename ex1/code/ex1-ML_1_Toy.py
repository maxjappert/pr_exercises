import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import math

from matplotlib.patches import Ellipse

dataPath = '../data/'

# Manual implementation of maximum likelihood for a 2D toy dataset

def calculate_sample_mean(data: np.ndarray) -> np.ndarray:
    # EXERCISE 2 - Compute the sample mean of a data array (2D vector output)
    d, n = np.shape(data)
    mean = np.zeros((d,))

    # Simple: The average of each data set is computed and added to the mean vector.

    for i in range(0, d):
        m = 0
        for j in range(0, n):
            m += data[i, j]
        mean[i] = m / n

    return mean


def calculate_sample_covariance_matrix(data: np.ndarray, mean: np.ndarray) -> np.ndarray:
    # EXERCISE 2 - Compute the sample covariance matrix

    # First we'll calculate the variances, which will fill the diagonal of the covariance matrix \Sigma.
    d, n = np.shape(data)
    variances = np.zeros((d,))

    for i in range(0, d):
        for j in range(0, n):
            variances[i] += pow(data[i, j] - mean[i], 2)
        variances[i] = variances[i] / (n-1)

    cov_matrix = np.zeros((d, d))

    for i in range(0, d):
        cov_matrix[i, i] = variances[i]

    # Now we'll calculate the covariance, whereof only one exists, because we're operating in 2D and \Sigma is
    # symmetric.
    covariance = 0

    for i in range(0, n):
        covariance += (data[0, i] - mean[0]) * (data[1, i] - mean[1])

    covariance = covariance / (n-1)

    # Add the covariance to the matrix.
    cov_matrix[0, 1] = covariance
    cov_matrix[1, 0] = covariance

    return cov_matrix


def matrix2dDeterminant(mat: np.ndarray) -> float:
    # EXERCISE 2 - Compute the determinant value of a 2x2 matrix
    # Just following the definition of the determinant of a 2D matrix.
    matrix_determinant = mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]

    return matrix_determinant


def matrix2dInverse(mat: np.ndarray) -> np.ndarray:
    # EXERCISE 2 - Compute the inverse matrix of a 2x2 matrix
    # Same as with the determinant. We just implement the formula for computing the inverse of a 2D matrix.
    matrix_inverse = np.zeros((2, 2))

    scalar = 1 / (mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0])

    matrix_inverse[0, 0] = scalar * mat[1, 1]
    matrix_inverse[0, 1] = -scalar * mat[0, 1]
    matrix_inverse[1, 0] = -scalar * mat[1, 0]
    matrix_inverse[1, 1] = scalar * mat[0, 0]

    return matrix_inverse


def pdf(x: np.array, mean: np.ndarray, cov: np.ndarray) -> float:
    # EXERCISE 2 - Implement PDF function for a 2D multivariate normal distribution (MVND)
    assert x.shape == mean.shape

    x_shape = x.shape

    d = x_shape[0]

    x_minus_mean = np.matrix(x - mean)

    # This variable is only introduced to make the code less convoluted. Otherwise we're simply implementing the
    # formula from the slides.
    potenz = float(((-1 / 2) * x_minus_mean * matrix2dInverse(cov) * x_minus_mean.T)[0, 0])

    probability = (1 / (pow(2 * math.pi, d) * matrix2dDeterminant(cov))) * pow(math.e, potenz)

    return probability


def classification(x: np.ndarray, mean_class0: np.ndarray, cov_class0: np.ndarray,
                   mean_class1: np.ndarray, cov_class1: np.ndarray) -> bool:
    # EXERCISE 2 - Implement classification function of a point into one of 2 MVND
    # Return True if class 1 and False if class 0.
    assigned_class = None

    p0 = pdf(x, mean_class0, cov_class0)
    p1 = pdf(x, mean_class1, cov_class1)

    # Just in case :)
    assert(0 <= p0 <= 1)
    assert(0 <= p1 <= 1)

    # The point is assigned to the class with the higher probability of the point belonging to it.
    if p0 > p1:
        assigned_class = False
    else:
        assigned_class = True

    return assigned_class


def box_muller_transform(unif1: float, unif2: float) -> (float, float):
    # EXERCISE 2 - Implement sampling from a standard normal distribution
    # Transforms 2 uniform samples into 2 random samples from a standard normal distribution N(0,1)
    # This implementation simply follows the formula from the exercise sheet.
    rnd1 = math.sqrt(-2 * math.log(unif1, math.e)) * math.cos(2 * math.pi * unif2)
    rnd2 = math.sqrt(-2 * math.log(unif1, math.e)) * math.sin(2 * math.pi * unif2)
    return rnd1, rnd2


def cholesky_factor_2d(mat: np.ndarray) -> np.ndarray:
    # EXERCISE 2 - Compute the cholesky decomposition of a 2x2 matrix
    d, n = np.shape(mat)
    # should only be applied to quadratic 2x2 matrices matrices
    # the matrix should be symmetric
    assert d == n and n == 2
    assert mat[0, 1] - mat[1, 0] < 1e-5

    L = np.zeros((2, 2))

    # This is just a simple implementation of the formula to compute the Cholesky Decomposition of a 2x2 Matrix.
    # The factors which allow for such a decomposition are checked above.

    L[0, 0] = math.sqrt(mat[0, 0])
    L[1, 0] = mat[0, 1] / L[0, 0]
    L[1, 1] = math.sqrt(mat[1, 1] - pow(L[1, 0], 2))

    return L


def sample_from_2d_gaussian(mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    # EXERCISE 2 - Implement sampling from MVND
    # Sample a random 2D vector from a normal distribution with given mean and covariance matrix
    # Step1: Sample two uniform random values
    random_number_uniform_1 = np.random.random()
    random_number_uniform_2 = np.random.random()
    # Step2: Transform the uniform samples into samples from a standard normal distribution
    # using the box_muller_transform
    (rnd1, rnd2) = box_muller_transform(random_number_uniform_1, random_number_uniform_2)

    sample = np.zeros((mean.shape))
    # Convert the normal samples into a 2D sample from a MVND with the provided mean and covariance matrix

    L = cholesky_factor_2d(cov)

    z = np.zeros((2,))
    z[0] = rnd1
    z[1] = rnd2

    sample = L * z + mean

    return sample


def print_classification(data: np.ndarray, labels: np.ndarray, class0_mean: np.ndarray, class0_cov: np.ndarray,
                         class1_mean: np.ndarray, class1_cov: np.ndarray) -> None:
    # Helper function to check if data items are classified correctly
    correctly_classified = 0
    false_class_0 = 0
    false_class_1 = 0

    _, n = np.shape(data)

    for i in range(n):
        if classification(data[:, i], class0_mean, class0_cov, class1_mean, class1_cov):
            if labels[i] == 1:
                correctly_classified += 1
            else:
                false_class_1 += 1

        else:
            if labels[i] == 0:
                correctly_classified += 1
            else:
                false_class_0 += 1

    print('############# DATA CLASSIFICATION #################')
    print('Classified', correctly_classified, 'out of', n, 'samples correctly.')
    print('False omega_1:', false_class_0)
    print('False omega_2:', false_class_1)
    print('###################################################\n')


def plot_cov(ax: plt.axes, mean: np.ndarray, cov_matrix: np.ndarray, color: str) -> None:
    # Helper function to visualize the distribution
    vals, vecs = np.linalg.eigh(cov_matrix)
    x, y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y, x))
    w, h = 2 * np.sqrt(vals)
    ax.add_artist(Ellipse(mean, w, h, theta, color=color, alpha=0.3))


if __name__ == '__main__':
    data_train = scio.loadmat(os.path.join(dataPath, 'mle_toy_train.mat'))
    data_test = scio.loadmat(os.path.join(dataPath, 'mle_toy_test.mat'))

    class0_training_data = data_train['omega_1']
    class0_training_labels = np.zeros((1, np.shape(class0_training_data)[1]))
    class1_training_data = data_train['omega_2']
    class1_training_labels = np.ones((1, np.shape(class1_training_data)[1]))

    class0_test_data = data_test['omega_1']
    class0_test_labels = np.zeros((1, np.shape(class0_test_data)[1]))
    class1_test_data = data_test['omega_2']
    class1_test_labels = np.ones((1, np.shape(class1_test_data)[1]))

    # visualize the data set
    fig1 = plt.figure(1)
    plt.plot(class0_training_data[0, :], class0_training_data[1, :], 'bx')
    plt.plot(class1_training_data[0, :], class1_training_data[1, :], 'r.')
    plt.legend(['$\omega_1$ Training', '$\omega_2$ Training'])
    plt.title('Training Data')
    plt.show(block=False)

    class0_mean = calculate_sample_mean(class0_training_data)
    class1_mean = calculate_sample_mean(class1_training_data)

    class0_cov = calculate_sample_covariance_matrix(class0_training_data, class0_mean)
    class1_cov = calculate_sample_covariance_matrix(class1_training_data, class1_mean)

    print("\nClass0: Estimated mean: {}, \nEstimated covariance matrix: \n{}".format(class0_mean, class0_cov))
    print("\nClass1: Estimated mean: {}, \nEstimated covariance matrix: \n{}".format(class1_mean, class1_cov))

    print("\nTraining data classification:")
    all_training_data = np.append(class0_training_data, class1_training_data, axis=1)
    all_training_labels = np.append(class0_training_labels, class1_training_labels)
    print_classification(all_training_data, all_training_labels, class0_mean, class0_cov, class1_mean, class1_cov)

    print("\nTest data classification:")
    all_test_data = np.append(class0_test_data, class1_test_data, axis=1)
    all_test_labels = np.append(class0_test_labels, class1_test_labels)
    print_classification(all_test_data, all_test_labels, class0_mean, class0_cov, class1_mean, class1_cov)


    class0_random_samples = []
    class1_random_samples = []

    for _ in range(10):
        class0_random_samples.append(sample_from_2d_gaussian(class0_mean, class0_cov).tolist())
        class1_random_samples.append(sample_from_2d_gaussian(class1_mean, class1_cov).tolist())

    # visualize estimated normal distributions
    fig2 = plt.figure(2)
    ax = plt.axes()
    plt.plot(class0_training_data[0, :], class0_training_data[1, :], 'bx')
    plt.plot(class0_test_data[0, :], class0_test_data[1, :], 'bo')
    plt.plot(list(zip(*class0_random_samples))[0], list(zip(*class0_random_samples))[1], 'k*')
    plot_cov(ax, class0_mean, class0_cov, 'blue')

    plt.plot(class1_training_data[0, :], class1_training_data[1, :], 'rx')
    plt.plot(class1_test_data[0, :], class1_test_data[1, :], 'ro')
    plt.plot(list(zip(*class1_random_samples))[0], list(zip(*class1_random_samples))[1], 'gD')
    plot_cov(ax, class1_mean, class1_cov, 'red')
    plt.legend(['$\omega_0$ Training', '$\omega_0$ Test', '$\omega_0$ Sample',
                '$\omega_1$ Training', '$\omega_1$ Test', '$\omega_1$ Sample'])
    plt.title('Training and test Data - TO UPLOAD on ADAM')
    plt.show()
