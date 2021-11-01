import math

import numpy as np


class LOGREG(object):
    '''
    Logistic regression class based on the LOGREG lecture slides
    '''

    def __init__(self, regularization: float = 0):
        self.r = regularization
        self._threshold = 10e-9
        self._eps = self._threshold

    def activationFunction(self, w: np.ndarray, X: np.ndarray) -> np.ndarray:
        # TODO: Implement logistic function

        #print(np.matrix(X).shape)
        d, n = X.shape

        result = np.ndarray(n)

        for i in range(0, n):
            #print(w.T.shape)
            #print(np.matrix(X[:, i]).shape)
            #print(w.T*np.matrix(X[:, i]).T + w[0])
            #print(np.matrix(w))
            #print(np.matrix(X[:, i]))
            #print(math.exp(-(np.matrix(w).T@np.matrix(X[:, i]).T)))
            result[i] = 1.0 / (1 + math.exp(-(np.matrix(w).T@np.matrix(X[:, i]).T)))

        # from slide 11
        return result

    def _costFunction(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        '''
        Compute the cost function for the current model parameters
        :param w: current model parameters
        :param X: data
        :param y: data labels
        :return: cost
        '''
        # TODO: Implement equation of cost function for posterior p(y=1|X,w)

        posterior = self.activationFunction(w, X)

        cost = 0

        regularizationTerm = 0

        for i in range (0, len(y)):
            if posterior[i] < self._eps:
                posterior[i] = self._eps

            cost += y[i] * math.log(posterior[i] / (1 - posterior[i])) + math.log(1 - posterior[i])

        return cost + regularizationTerm

    def _calculateDerivative(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        Compute the derivative of the model parameters
        :param w: current model parameters
        :param X: data
        :param y: data labels
        :return: first derivative of the model parameters
        '''
        # TODO: Calculate derivative of loglikelihood function for posterior p(y=1|X,w)
        firstDerivative = np.zeros(len(y))
        regularizationTerm = 0

        sigmoid = self.activationFunction(w, X)

        for i in range(0, len(y)):
            firstDerivative += (y[i] - sigmoid[i])@X[:, i]

        return firstDerivative + regularizationTerm

    def _calculateHessian(self, w: np.ndarray, X: np.ndarray) -> np.ndarray:
        '''
        :param w: current model parameters
        :param X: data
        :return: the hessian matrix (second derivative of the model parameters)
        '''
        # TODO: Calculate Hessian matrix of loglikelihood function for posterior p(y=1|X,w)

        n, d = X.shape

        hessian = np.zeros((n, n))
        regularizationTerm = 0

        sigma = self.activationFunction(np.matrix(w), X)

        for i in range (0, n):
            hessian += np.matrix(X[:, i]).T@np.matrix(X[:, i])*sigma[i]*(1 - sigma[i])
            assert sigma[i] * (1 - sigma[i]) >= 0

        return - hessian + regularizationTerm

    def _optimizeNewtonRaphson(self, X: np.ndarray, y: np.ndarray, number_of_iterations: int) -> np.ndarray:
        '''
        Newton Raphson method to iteratively find the optimal model parameters (w)
        :param X: data
        :param y: data labels (0 or 1)
        :param number_of_iterations: number of iterations to take
        :return: model parameters (w)
        '''
        # TODO: Implement Iterative Reweighted Least Squares algorithm for optimization, use the calculateDerivative and calculateHessian functions you have already defined above
        w = np.zeros((X.shape[0], 1))

        posteriorloglikelihood = self._costFunction(w, X, y)
        print('initial posteriorloglikelihood', posteriorloglikelihood, 'initial likelihood',
              np.exp(posteriorloglikelihood))

        for i in range(number_of_iterations):
            oldposteriorloglikelihood = posteriorloglikelihood
            w_old = w
            h = self._calculateHessian(w, X)
            #sigma = self.activationFunction(w_old, X)

            w = w_old - np.linalg.inv(h)@self._calculateDerivative(w_old, X, y)
            print(w_old.shape)
            print(w.shape)
            w_update = w - w_old
            posteriorloglikelihood = self._costFunction(w, X, y)
            if self.r == 0:
                # TODO: What happens if this condition is removed?
                if np.exp(posteriorloglikelihood) > 1 - self._eps:
                    print('posterior > 1-eps, breaking optimization at niter = ', i)
                    break

            # TODO: Implement convergence check based on when w_update is close to zero
            # Note: You can make use of the class threshold value self._threshold

            for j in range(0, len(w)):
                if abs(w_update[j]) < self._eps:
                    break

        print('final posteriorloglikelihood', posteriorloglikelihood, 'final likelihood',
              np.exp(posteriorloglikelihood))

        # Note: maximize likelihood (should become larger and closer to 1), maximize loglikelihood( should get less negative and closer to zero)
        return w

    def train(self, X: np.ndarray, y: np.ndarray, iterations: int) -> np.ndarray:
        '''
        :param X: dataset
        :param y: ground truth labels
        :param iterations: Number of iterations to train
        :return: trained w parameter
        '''
        self.w = self._optimizeNewtonRaphson(X, y, iterations)
        return self.w

    def classify(self, X: np.ndarray) -> np.ndarray:
        '''
        Classify data given the trained logistic regressor - access the w parameter through self.
        :param x: Data to be classified
        :return: List of classification values (0.0 or 1.0)
        '''
        # TODO: Implement classification function for each entry in the data matrix
        numberOfSamples = X.shape[1]

        predictions = np.zeros(numberOfSamples)

        for i in range(0, numberOfSamples):
            predictions[i] = 0 if np.matrix(self.w).T@X[:, i] + self.w[0] < 0 else 1

        return predictions

    def printClassification(self, X: np.ndarray, y: np.ndarray) -> None:
        '''
        Calls "classify" and computes the total classification error given the ground truth labels
        :param x: Data to be classified
        :param y: Ground truth labels
        '''
        # TODO: Implement print classification
        numberOfSamples = X.shape[1]

        predictions = self.classify(X)

        numOfMissclassified = 0.0

        for i in range(0, numberOfSamples):
            if predictions[i] != y[i]:
                numOfMissclassified += 1

        totalError = numOfMissclassified / numberOfSamples

        print("{}/{} misclassified. Total error: {:.2f}%.".format(numOfMissclassified, numberOfSamples, totalError))
