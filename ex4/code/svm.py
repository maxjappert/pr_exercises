import numpy as np
from scipy.linalg import norm
import cvxopt as cvx


class SVM(object):
    '''
    SVM class
    '''

    def __init__(self, C=None):
        self.C = C
        self.__TOL = 1e-5

    def __linearKernel__(self, x1: np.ndarray, x2: np.ndarray, _) -> float:
        # TODO: Implement linear kernel function
        # @x1 and @x2 are vectors
        return np.transpose(x1).dot(x2)

    def __polynomialKernel__(self, x1: np.ndarray, x2: np.ndarray, p: int) -> float:
        # TODO: Implement polynomial kernel function
        # @x1 and @x2 are vectors
        return np.power((np.transpose(x1).dot(x2) + 1), 2)

    def __gaussianKernel__(self, x1: np.ndarray, x2: np.ndarray, sigma: float) -> float:
        # TODO: Implement gaussian kernel function
        # @x1 and @x2 are vectors
        return np.exp(- ((np.linalg.norm(x1 - x2, 2))**2 / (sigma**2)))

    def __computeKernelMatrix__(self, x: np.ndarray, kernelFunction, pars) -> np.ndarray:
        # TODO: Implement function to compute the kernel matrix
        # @x is the data matrix
        # @kernelFunction - pass a kernel function (gauss, poly, linear) to this input
        # @pars - pass the possible kernel function parameter to this input

        # We have no idea what we're doing...

        # the matrix needs to be n x n
        K = np.zeros((x.shape[1], x.shape[1]))

        print(kernelFunction)

        if kernelFunction == "linear":
            for i in range(0, x.shape[1]):
                for j in range(0, x.shape[1]):
                    K[i, j] = self.__linearKernel__(x[:, i], x[:, j], pars)
        elif kernelFunction == "poly":
            for i in range(0, x.shape[1]):
                for j in range(0, x.shape[1]):
                    K[i, j] = self.__polynomialKernel__(x[:, i], x[:, j], pars)
        elif kernelFunction == "rbf":
            for i in range(0, x.shape[1]):
                for j in range(0, x.shape[1]):
                    K[i, j] = self.__gaussianKernel__(x[:, i], x[:, j], pars)
        else:
            print("error computing kernel matrix:", kernelFunction)


        return K

    def train(self, x: np.ndarray, y: np.ndarray, kernel=None, kernelpar=2) -> None:
        # TODO: Implement the remainder of the svm training function
        self.kernelpar = kernelpar

        NUM = x.shape[1]

        # we'll solve the dual
        # obtain the kernel
        if kernel == 'linear':
            # Compute the kernel matrix for the non-linear SVM with a linear kernel
            print('Fitting SVM with linear kernel')
            self.kernel = self.__linearKernel__
            K = self.__computeKernelMatrix__(x, "linear", kernelpar)
        elif kernel == 'poly':
            # Compute the kernel matrix for the non-linear SVM with a polynomial kernel
            print('Fitting SVM with Polynomial kernel, order: {}'.format(kernelpar))
            self.kernel = self.__polynomialKernel__
            K = self.__computeKernelMatrix__(x, "poly", kernelpar)
        elif kernel == 'rbf':
            # Compute the kernel matrix for the non-linear SVM with an RBF kernel
            print('Fitting SVM with RBF kernel, sigma: {}'.format(kernelpar))
            self.kernel = self.__gaussianKernel__
            K = self.__computeKernelMatrix__(x, "rbf", kernelpar)
        else:
            print('Fitting linear SVM')
            # Compute the kernel matrix for the linear SVM
            K = self.__computeKernelMatrix__(x, "linear", kernelpar)
            self.kernel = self.__linearKernel__

        # We found these values by trial and error
        if self.C is None:
            G = -np.eye(NUM)
            h = np.zeros((NUM, 1))
            print("No C value specified")
        else:
            print("Using Slack variables")
            print("C = ", self.C)
            G = np.concatenate((- np.eye(NUM), np.eye(NUM)))
            h = np.concatenate((np.zeros((NUM, 1)), self.C * np.ones((NUM, 1))))

        # Compute below values according to the lecture slides
        cvx.solvers.options["show_progress"] = False

        # We admittedly don't know why this line is necessary, we never found a formula which describes
        # it. It is only here because we discussed our at that point not-working solution with others, whereby we
        # were recommended to add this line and that made it work. Just in case it pops up when checking for plagiarism.
        K = np.multiply(K, np.transpose(y).dot(y))

        P = cvx.matrix(K)
        q = cvx.matrix(-np.ones((NUM, 1)))
        G = cvx.matrix(G)
        h = cvx.matrix(h)
        A = cvx.matrix(y)
        b = cvx.matrix(np.zeros(1))

        # As on the exercise sheet
        solution = cvx.solvers.qp(P, q, G, h, A, b)

        # All the lambdas, including those which are zero
        all_lambdas = np.array(solution['x'])

        # The larger_than_zero array has a 1 where lambda is larger than 0 and 0 otherwise.
        # self.indices then saves the indices of larger_than_zero of all the 1s
        # These indices then correspond to support vectors in x
        larger_than_zero = abs(all_lambdas >= self.__TOL)
        self.indices = np.where(larger_than_zero)[0]

        self.lambdas = all_lambdas[self.indices, 0] # Only save > 0
        self.sv = np.transpose(x[:, self.indices])
        self.sv_labels = y[0, self.indices]

        if kernel is None:
            self.w = np.zeros(x.shape[0]) # SVM weights used in the linear SVM

            for i in range(0, len(self.sv_labels)):
                self.w += self.lambdas[i] * self.sv_labels[i] * self.sv[i, :]

            self.bias = np.mean(self.sv_labels - np.dot(self.w, np.transpose(self.sv)))
        else:
            self.w = None
            # Use the mean of all support vectors for stability when computing the bias (w_0).
            # In the kernel case, remember to compute the inner product with the chosen kernel function.

            # TODO: Change to something more original!
            self.bias = np.mean(y[:, self.indices] - np.multiply(y[:, self.indices], np.multiply(self.lambdas, np.sum(self.__computeKernelMatrix__(np.transpose(self.sv), kernel, self.kernelpar), axis=0))))

        # Implement the KKT check
        self.__check__()

    def __check__(self) -> None:
        # Checking implementation according to KKT2 (Linear_classifiers slide 46)
        # Slide 46 doesn't exist. We're using the formula from slide 21, whereby we're using the
        # the sv_labels instead of y because for each y_i it holds that y_i * lambda_i == 0 if lambda_i == 0
        kkt2_check = np.sum(self.lambdas * self.sv_labels)

        assert kkt2_check < self.__TOL, 'SVM check failed - KKT2 condition not satisfied'

    def classifyLinear(self, x: np.ndarray) -> np.ndarray:
        '''
        Classify data given the trained linear SVM - access the SVM parameters through self.
        :param x: Data to be classified
        :return: List of classification values (-1.0 or 1.0)
        '''
        # Implement

        classificationFuncion = np.zeros(x.shape[1])

        # This could be done a lot more efficiently.
        for i in range(0, x.shape[1]):
            a = np.matrix(self.w)
            b = np.transpose(np.matrix(x[:, i]))
            c = np.dot(a, b)
            classificationFuncion[i] = c[0, 0] + self.bias

        return np.where(classificationFuncion < 0, -1, 1)

    def printLinearClassificationError(self, x: np.ndarray, y: np.ndarray) -> None:
        '''
        Calls classifyLinear and computes the total classification error given the ground truth labels
        :param x: Data to be classified
        :param y: Ground truth labels
        '''
        # Implement

        classifiedLabels = self.classifyLinear(x)

        y_list = np.squeeze(np.asarray(y))

        result = (np.sum(np.where(classifiedLabels == y_list, 0, 1)) / len(y_list)) * 100

        print("Total error: {:.2f}%".format(result))

    def classifyKernel(self, x: np.ndarray) -> np.ndarray:
        '''
        Classify data given the trained kernel SVM - use self.kernel and self.kernelpar to access the kernel function and parameter
        :param x: Data to be classified
        :return: List of classification values (-1.0 or 1.0)
        '''
        # Implement

        classification_function = np.ones(x.shape[1]) * self.bias

        #for i in range(0, len(self.indices)):
        #    classification_function += self.lambdas * self.sv_labels * self.kernel(self.sv[i], np.sum(x, axis=0), self.kernelpar)

        for j in range(0, x.shape[1]):
            for i in range(0, len(self.indices)):
                classification_function[j] += self.lambdas[i] * self.sv_labels[i] * self.kernel(self.sv[i], x[:, j], self.kernelpar)

        #classification_function = np.sum(self.lambdas * self.sv_labels * self.kernel(np.transpose(np.matrix(self.sv)), x, self.kernelpar), axis=0)

        return np.where(classification_function < 0, -1, 1)

    def printKernelClassificationError(self, x: np.ndarray, y: np.ndarray) -> None:
        '''
        Calls classifyKernel and computes the total classification error given the ground truth labels
        :param x: Data to be classified
        :param y: Ground truth labels
        '''

        # TODO: Implement

        labels = self.classifyKernel(x)

        y_list = np.squeeze(np.asarray(y))

        assert(len(labels) == len(y_list))

        result = (np.sum(np.where(labels == y_list, 0, 1)) / len(labels)) * 100
        print("Total error: {:.2f}%".format(result))
