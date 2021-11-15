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
        return np.power((np.transpose(x1).dot(x2) + 1), p)

    def __gaussianKernel__(self, x1: np.ndarray, x2: np.ndarray, sigma: float) -> float:
        # TODO: Implement gaussian kernel function
        # @x1 and @x2 are vectors
        return np.exp(- np.linalg.norm(x1 - x2, 2)**2 / sigma**2)

    def __computeKernelMatrix__(self, x: np.ndarray, kernelFunction, pars) -> np.ndarray:
        # TODO: Implement function to compute the kernel matrix
        # @x is the data matrix
        # @kernelFunction - pass a kernel function (gauss, poly, linear) to this input
        # @pars - pass the possible kernel function parameter to this input

        # We have no idea what we're doing...

        # the matrix needs to be n x n
        K = np.zeros((x.shape[1], x.shape[1]))

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
            print("error computing kernel matrix")

        return K

    def train(self, x: np.ndarray, y: np.ndarray, kernel=None, kernelpar=2) -> None:
        # TODO: Implement the remainder of the svm training function
        self.kernelpar = kernelpar

        NUM = x.shape[1]

        # we'll solve the dual
        # obtain the kernel
        if kernel == 'linear':
            # TODO: Compute the kernel matrix for the non-linear SVM with a linear kernel
            print('Fitting SVM with linear kernel')
            self.kernel = self.__linearKernel__
            K = self.__computeKernelMatrix__(x, "linear", kernelpar)
        elif kernel == 'poly':
            # TODO: Compute the kernel matrix for the non-linear SVM with a polynomial kernel
            print('Fitting SVM with Polynomial kernel, order: {}'.format(kernelpar))
            self.kernel = self.__polynomialKernel__
            K = self.__computeKernelMatrix__(x, "poly", kernelpar)
        elif kernel == 'rbf':
            # TODO: Compute the kernel matrix for the non-linear SVM with an RBF kernel
            print('Fitting SVM with RBF kernel, sigma: {}'.format(kernelpar))
            self.kernel = self.__gaussianKernel__
            K = self.__computeKernelMatrix__(x, "rbf", kernelpar)
        else:
            print('Fitting linear SVM')
            # TODO: Compute the kernel matrix for the linear SVM
            #K = np.zeros((NUM, NUM))
            K = self.__computeKernelMatrix__(x, "linear", kernelpar)
            self.kernel = self.__linearKernel__

            # for i in range(0, NUM):
            #     for j in range(0, NUM):
            #         K[i, j] = y[0, i] * y[0, j] * (np.matrix(x[:, i])@np.transpose(np.matrix(x[:, j])))

        # This probably doesn't work tbh
        if self.C is None:
            G = -np.eye(NUM)
            h = np.zeros((NUM, 1))
            print("No C value specified")
        else:
            print("Using Slack variables")
            print("C = ", self.C)
            G = np.concatenate((- np.eye(NUM), np.eye(NUM)))
            h = np.concatenate((np.zeros((NUM, 1)), self.C * np.ones((NUM, 1))))

        # TODO: Compute below values according to the lecture slides
        cvx.solvers.options["show_progress"] = False

        P = cvx.matrix(K)
        q = cvx.matrix(-np.ones((NUM, 1)))
        G = cvx.matrix(G)
        h = cvx.matrix(h)
        A = cvx.matrix(y)
        b = cvx.matrix(np.zeros(1))

        # TODO: Change this to something more original!
        solution = cvx.solvers.qp(P, q, G, h, A, b)
        filter = abs(np.array(solution['x']) >= self.__TOL)
        self.indices = np.where(filter)[0]

        self.lambdas = np.zeros(len(self.indices)) # Only save > 0
        self.sv = np.zeros((x.shape[0], len(self.indices)))
        self.sv_labels = np.zeros(len(self.indices))

        i = 0
        for index in self.indices:
            self.lambdas[i] = np.array(solution['x'])[index]
            self.sv[:, i] = x[:, index]
            self.sv_labels[i] = y[0, index]
            i += 1

        print(self.lambdas)


        if kernel is None:
            self.w = np.zeros(x.shape[0]) # SVM weights used in the linear SVM
            checksum = 0

            for j in range(0, x.shape[1]):
                self.w += np.array(solution['x'])[j] * x[:, j] * y[0, j]
                checksum += np.array(solution['x'][j]) * y[0, j]


            # slide 25
            assert np.abs(checksum <= self.__TOL)

            #w0s = np.zeros(len(self.indices))


            # This would be the formula but the classification becomes useless when we use this rather than
            # bias = 0

            #for i in range(0, len(self.indices)):
            #    w0s[i] = (1.0 / y[0, self.indices[i]] - (np.matrix(self.w).T@np.matrix(x[:, self.indices[i]])))[0, 0]

            # Use the mean of all support vectors for stability when computing the bias (w_0)
            #self.bias = np.sum(w0s) / len(w0s) # Bias

            #print("Bias:", self.bias)

            self.bias = 0

        else:
            self.w = None
            # Use the mean of all support vectors for stability when computing the bias (w_0).
            # In the kernel case, remember to compute the inner product with the chosen kernel function.

            self.bias = self.__TOL


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

        #classificationFuncion = np.matrix(self.w).T@x + self.bias

        classificationFuncion = np.zeros(x.shape[1])

        # TODO: This could be done a lot more efficiently

        for i in range(0, x.shape[1]):
            a = np.matrix(self.w)
            b = np.transpose(np.matrix(x[:, i]))
            c = a@b
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

        #assert(len(y) == len(classifiedLabels))

        y_list = np.squeeze(np.asarray(y))

        result = np.sum(np.where(classifiedLabels == y_list, 0, 1)) / len(y_list)

        print("Total error: {:.2f}%".format(result))

    def classifyKernel(self, x: np.ndarray) -> np.ndarray:
        '''
        Classify data given the trained kernel SVM - use self.kernel and self.kernelpar to access the kernel function and parameter
        :param x: Data to be classified
        :return: List of classification values (-1.0 or 1.0)
        '''
        # Implement

        classificationFunction = np.ones(x.shape[1]) * self.bias

        # TODO: this is so inefficient!

        for j in range(0, x.shape[1]):
            for i in range(0, len(self.indices)):
                classificationFunction[i] += self.lambdas[i] * self.sv_labels[i] * self.kernel(x[:, self.indices[i]], x[:, j], self.kernelpar)

        return np.where(classificationFunction < 0, -1, 1)

    def printKernelClassificationError(self, x: np.ndarray, y: np.ndarray) -> None:
        '''
        Calls classifyKernel and computes the total classification error given the ground truth labels
        :param x: Data to be classified
        :param y: Ground truth labels
        '''
        result = 0

        # TODO: Implement

        labels = self.classifyKernel(x)

        y_list = np.squeeze(np.asarray(y))

        assert(len(labels) == len(y_list))

        result = (np.sum(np.where(labels == y_list, 0, 1)) / len(labels)) * 100

        cardClass1 = 0
        cardClass2 = 0

        for i in range(len(labels)):
            if labels[i] == -1:
                cardClass1 += 1
            elif labels[i] == 1:
                cardClass2 += 1
            else:
                print("Weird!")

        #print("Total points:", len(labels))
        #print("Total misclassified:", np.sum(np.where(labels == y_list, 0, 1)))
        print("Classified into class 1", cardClass1)
        print("Classified into class 2", cardClass2)
        print("Total error: {:.2f}%".format(result))
