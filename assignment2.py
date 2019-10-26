#!/usr/bin/env python3 -u

import numpy as np
import sklearn.datasets as skdata
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

import argparse
import logging

from pprint import pprint
from pprint import pformat

import re

# Parse command line options
parser = argparse.ArgumentParser(description='Do stuff.')
parser.add_argument(
    '-d', '--debug', dest='debug', action='store_true', default=0, help='debug mode')
parser.add_argument(
    '-i', '--info', dest='info', action='store_true', default=0, help='info mode')
parser.add_argument(
    '-w', '--warn', dest='warn', action='store_true', default=0, help='warn mode')
parser.add_argument(
    '-tc', '--t_cancer', dest='t_cancer', type=int, default=100)
parser.add_argument(
    '-ac', '--alpha_cancer', dest='alpha_cancer', type=float, default=1e-4)
parser.add_argument(
    '-ec', '--epsilon_cancer', dest='epsilon_cancer', type=float, default=1e-8)

args = parser.parse_args()

# Initialize logging
# logging.basicConfig(format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
logging.basicConfig(format='%(levelname)s [%(filename)s:%(lineno)4d %(funcName)s] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

log = logging.getLogger(__name__)
if args.debug:
    log.setLevel(logging.DEBUG)
elif args.info:
    log.setLevel(logging.INFO)
elif args.warn:
    log.setLevel(logging.WARNING)

t_cancer = args.t_cancer
alpha_cancer = args.alpha_cancer
epsilon_cancer = args.epsilon_cancer

"""
Name: Doe, John (Please write names in <Last Name, First Name> format)

Collaborators: Doe, Jane (Please write names in <Last Name, First Name> format)

Collaboration details: Discussed <function name> implementation details with Jane Doe.

Summary:
Report your scores here. For example,

Results on Wisconsin breast cancer dataset using scikit-learn Logistic Regression model
Training set mean accuracy: 0.9590
Testing set mean accuracy: 0.9649
Results on digits 7 and 9 dataset using scikit-learn Logistic Regression model
Training set mean accuracy: 1.0000
Testing set mean accuracy: 0.9722
Results on Boston housing price dataset using scikit-learn Linear Regression model
Training set mean accuracy: 23.2335
Testing set mean accuracy: 10.8062
Results on diabetes dataset using scikit-learn Linear Regression model
Training set mean accuracy: 2991.9850
Testing set mean accuracy: 1735.9381
Results on Wisconsin breast cancer dataset using our Logistic Regression model
Training set mean accuracy: 0.0000
Testing set mean accuracy: 0.0000
Results on digits 7 and 9 dataset using our Logistic Regression model
Training set mean accuracy: 0.0000
Testing set mean accuracy: 0.0000
Results on Boston housing price dataset using our Linear Regression model
Training set mean accuracy: 617.1287
Testing set mean accuracy: 369.2698
Results on diabetes dataset using our Linear Regression model
Training set mean accuracy: 29088.9673
Testing set mean accuracy: 28946.6889
"""


"""
Implementation of our Gradient Descent optimizer for mean squared loss and logistic loss
"""


class GradientDescentOptimizer(object):
    def __init__(self):
        pass

    def __compute_gradients(self, w, x, y, loss_func):
        """
        Returns the gradient of the logistic, mean squared or half mean squared loss

        x : N x d feature vector
        y : N x 1 ground-truth label
        loss_func : loss type either 'logistic','mean_squared', or 'half_mean_squared'

        returns 1 x d gradients
        """

        # Prepend the set of x attributes with 0.5 bias (in Perceptron, this is threshold)
        bias = 0.5 * np.ones([x.shape[0], 1])

        # Concatenate the bias above
        x = np.concatenate([bias, x], axis=-1)  # x is now [N, d + 1]

        if loss_func == 'logistic':

            gradients = np.zeros(x.shape)
            for n in range(x.shape[0]):
                x_n = x[n, ...]

                # log.info("x_n.shape: " + str(x_n.shape))
                # log.info("w.shape: " + str(np.squeeze(w).shape))

                h_x = np.dot(np.squeeze(w), x_n)

                # Gradient Descent, slide 22
                gradients[n, :] = (-y[n] * x_n) / (1.0 + np.exp(y[n] * h_x))

                # there are N gradients

            return np.mean(gradients, axis=0)  # How does this match the slide 22?

        elif loss_func == 'mean_squared':
            return 0.0
        elif loss_func == 'half_mean_squared':
            return 0.0
        else:
            raise ValueError('Supported losses: logistic, mean_squared, or half_mean_squared')

    def update(self, w, x, y, alpha, loss_func):
        """
        Updates the weight vector based on logistic, mean squared or half mean squared loss

        w : 1 x d weight vector
        x : N x d feature vector
        y : N x 1 ground-truth label
        alpha : learning rate
        loss_func : loss type either 'logistic','mean_squared', or 'half_mean_squared'

        returns 1 x d weights
        """

        # Call compute_gradients (Gradient Descent, slide 18)
        w = w - alpha * self.__compute_gradients(w, x, y, loss_func)

        return w


"""
Implementation of our Logistic Regression model for binary classification
trained using Gradient Descent
"""


class LogisticRegressionGradientDescent(object):
    def __init__(self):
        # Define private variables
        self.__weights = None
        self.__optimizer = GradientDescentOptimizer()

    def fit(self, x, y, t, alpha, epsilon):
        """
        Fits the model to x and y by updating the weight vector
        using gradient descent

        x : N x d feature vector
        y : N x 1 ground-truth label
        t : number of iterations to train
        alpha : learning rate
        epsilon : threshold for stopping condition
        """

        # +1 is the threshold
        self.__weights = np.zeros([1, x.shape[1]+1])
        self.__weights[0] = -1

        for i in range(int(t)):
            # log.info("LOGISTIC CANCER FIT LOOP i=" + str(i))

            # predict
            h_x = self.predict(x)
            # log.info("h_x: " + str(h_x))  # vector

            h_x_mag = np.sqrt(np.sum(h_x ** 2))
            # log.info("h_x_mag: " + str(h_x_mag))  # vector

            # compute the loss (Gradient Descent, slide 21)
            # loss = np.mean((h_x_y) ** 2)
            loss = np.mean(np.log(1 + np.exp(-y * h_x)))  # (N, 1) and (N, 1)

            # call update (which calls compute the gradients)
            w_i = self.__optimizer.update(self.__weights, x, y, alpha, 'logistic')
            # log.info("w_i: " + str(w_i)) # vector
            # log.info("weights: " + str(self.__weights))

            if loss == 0:
                break

            # d_w is the change in weights
            d_w = self.__weights - w_i
            # log.info("d_w: " + str(d_w))  # vector

            d_w_mag = np.sqrt(np.sum(d_w ** 2))
            # log.info("d_w_mag: " + str(d_w_mag))

            # magnitude of the change in weights
            mag = np.sqrt(np.sum(d_w ** 2))
            # log.info("eps: " + str(epsilon))

            log.info("i=" + str(i) + " loss=" + str(loss) + " mag=" + str(mag))

            # Save the new weights
            self.__weights = w_i

            # check stopping conditions
            if mag < epsilon:  # epsilon is expected mag when done
                log.info("mag: " + str(mag))
                log.info("eps: " + str(epsilon))
                break

    def predict(self, x):
        """
        Predicts the label for each feature vector x

        x : N x d feature vector

        returns : N x 1 label vector
        """

        # Prepend the set of x attributes with 0.5 bias (in Perceptron, this is threshold)
        bias = 0.5 * np.ones([x.shape[0], 1])

        # Concatenate the bias above
        x = np.concatenate([bias, x], axis=-1)  # x is now [N, d + 1]

        # Prepare an array for predictions
        predictions = np.zeros(x.shape[0])

        # Iterate through every row in x
        for n in range(x.shape[0]):
            x_n = x[n, ...]

            h_x = np.dot(np.squeeze(self.__weights.T), x_n)
            predictions[n] = 1 / (1 + np.exp(-1 * h_x))  # Logistic Reg, slide 11

        predictions = np.where(predictions >= 0.5, 1.0, -1.0)

        return predictions

        # return np.zeros(x.shape[0])

    def score(self, x, y):
        """
        Predicts labels based on feature vector x and computes the mean accuracy
        of the predictions

        x : N x d feature vector
        y : N x 1 ground-truth label

        returns : double
        """
        h_x = self.predict(x)

        return np.mean(np.where(h_x == y, 1.0, 0.01))


"""
Implementation of our Linear Regression model trained using Gradient Descent
"""


class LinearRegressionGradientDescent(object):
    def __init__(self):
        # Define private variables
        self.__weights = None
        self.__optimizer = GradientDescentOptimizer()

    def fit(self, x, y, t, alpha, epsilon):
        """
        Fits the model to x and y by updating the weight vector
        using gradient descent

        x : N x d feature vector
        y : N x 1 ground-truth label
        t : number of iterations to train
        alpha : learning rate
        epsilon : threshold for stopping condition
        """

        # Initialize weights
        self.__weights = np.zeros([1, x.shape[1] + 1])
        self.__weights[0] = -1.0

        prev_w = self.__weights
        prev_loss = np.inf  # positive infinity

        # for i in range(t):
        #     # predict
        #     predictions = self.predict(x)

        #     # compute the loss
        #     # loss = np.mean((h_x_y) ** 2)

        #     # compute the gradients

        #     # check stopping conditions

    def predict(self, x):
        """
        Predicts the label for each feature vector x

        x : N x d feature vector

        returns : N x 1 label vector
        """

        # what should weights be equal to?

        x = np.concatenate([0.5 * np.ones([x.shape[0], 1]), x], axis=-1)

        log.info("x.shape: " + str(x.shape))

        h_x = np.zeros(x.shape[0])
        log.info("h_x.shape: " + str(h_x.shape))

        for n in range(x.shape[0]):
            x_n = x[n, ...]
            x_n = np.expand_dims(x_n, axis=-1)
            # h_x[n] = np.dot(self.__weights.T, x_n.T)

            h_x[n] = np.dot(np.squeeze(self.__weights), x_n)

        # log.info("h_x: " + str(h_x))

        return h_x

    def score(self, x, y):
        """
        Predicts labels based on feature vector x and computes the
        mean squared loss of the predictions

        x : N x d feature vector
        y : N x 1 ground-truth label

        returns : double
        """

        # Weight [1 x d+1]
        # MSE = 1/N

        # Take the exam
        h_x = self.predict(x)  # (N x 1)

        # Grade the exam
        return np.mean(y - h_x) ** 2


def mean_squared_error(y_hat, y):  # Gradient Descent, slide 15
    """
    Computes the mean squared error

    y_hat : N x 1 predictions
    y : N x 1 ground-truth label

    returns : double
    """
    return np.mean((y_hat-y)**2)


if __name__ == '__main__':

    # Loads breast cancer data with 90% training, 10% testing split
    breast_cancer_data = skdata.load_breast_cancer()
    x_cancer = breast_cancer_data.data
    y_cancer = breast_cancer_data.target

    split_idx = int(0.90*x_cancer.shape[0])
    x_cancer_train, y_cancer_train = x_cancer[:split_idx, :], y_cancer[:split_idx]
    x_cancer_test, y_cancer_test = x_cancer[split_idx:, :], y_cancer[split_idx:]

    # Loads 7 and 9 from digits data with 90% training, 10% testing split
    digits_data = skdata.load_digits()
    x_digits = digits_data.data
    y_digits = digits_data.target

    idx_79 = np.where(np.logical_or(y_digits == 7, y_digits == 9))[0]
    x_digits79 = x_digits[idx_79, :]
    y_digits79 = y_digits[idx_79]

    split_idx = int(0.90*x_digits79.shape[0])
    x_digits79_train, y_digits79_train = x_digits79[:split_idx, :], y_digits79[:split_idx]
    x_digits79_test, y_digits79_test = x_digits79[split_idx:, :], y_digits79[split_idx:]

    # Loads Boston housing price data with 90% training, 10% testing split
    housing_data = skdata.load_boston()
    x_housing = housing_data.data
    y_housing = housing_data.target

    split_idx = int(0.90*x_housing.shape[0])
    x_housing_train, y_housing_train = x_housing[:split_idx, :], y_housing[:split_idx]
    x_housing_test, y_housing_test = x_housing[split_idx:, :], y_housing[split_idx:]

    # Loads diabetes data with 90% training, 10% testing split
    diabetes_data = skdata.load_diabetes()
    x_diabetes = diabetes_data.data
    y_diabetes = diabetes_data.target

    split_idx = int(0.90*x_diabetes.shape[0])
    x_diabetes_train, y_diabetes_train = x_diabetes[:split_idx, :], y_diabetes[:split_idx]
    x_diabetes_test, y_diabetes_test = x_diabetes[split_idx:, :], y_diabetes[split_idx:]

    # """ SCI KIT LEARN LOGISTIC
    # Trains and tests Logistic Regression model from scikit-learn
    # """
    # Trains scikit-learn Logistic Regression model on Wisconsin cancer data
    scikit_logistic_cancer = LogisticRegression(solver='liblinear', verbose=2)
    scikit_logistic_cancer.fit(x_cancer_train, y_cancer_train)
    print('Results on Wisconsin breast cancer dataset using scikit-learn Logistic Regression model')
    # Test model on training set
    scikit_scores_cancer_train = scikit_logistic_cancer.score(x_cancer_train, y_cancer_train)
    print('Training set mean accuracy: {:.4f}'.format(scikit_scores_cancer_train))
    # Test model on testing set
    scikit_scores_cancer_test = scikit_logistic_cancer.score(x_cancer_test, y_cancer_test)
    print('Testing set mean accuracy: {:.4f}'.format(scikit_scores_cancer_test))

    params = scikit_logistic_cancer.get_params()
    log.info("params: " + str(params))
    pprint(params)
    log.info(pformat(params))

    # # Trains scikit-learn Logistic Regression model on digits 7 and 9 data
    # scikit_logistic_digits79 = LogisticRegression(solver='liblinear')
    # scikit_logistic_digits79.fit(x_digits79_train, y_digits79_train)
    # print('Results on digits 7 and 9 dataset using scikit-learn Logistic Regression model')
    # # Test model on training set
    # scikit_scores_digits79_train = scikit_logistic_digits79.score(
    #     x_digits79_train, y_digits79_train)
    # print('Training set mean accuracy: {:.4f}'.format(scikit_scores_digits79_train))
    # # Test model on testing set
    # scikit_scores_digits79_test = scikit_logistic_digits79.score(x_digits79_test, y_digits79_test)
    # print('Testing set mean accuracy: {:.4f}'.format(scikit_scores_digits79_test))

    # """ SCI KIT LEARN LINEAR
    # Trains and tests Linear Regression model from scikit-learn
    # """
    # # Trains scikit-learn Linear Regression model on Boston housing price data
    # scikit_linear_housing = LinearRegression()
    # scikit_linear_housing.fit(x_housing_train, y_housing_train)
    # print('Results on Boston housing price dataset using scikit-learn Linear Regression model')
    # # Test model on training set
    # scikit_predictions_housing_train = scikit_linear_housing.predict(x_housing_train)
    # scikit_scores_housing_train = mean_squared_error(
    #     scikit_predictions_housing_train, y_housing_train)
    # print('Training set mean accuracy: {:.4f}'.format(scikit_scores_housing_train))
    # # Test model on testing set
    # scikit_predictions_housing_test = scikit_linear_housing.predict(x_housing_test)
    # scikit_scores_housing_test = mean_squared_error(scikit_predictions_housing_test, y_housing_test)
    # print('Testing set mean accuracy: {:.4f}'.format(scikit_scores_housing_test))

    # # Trains scikit-learn Linear Regression model on diabetes data
    # scikit_linear_diabetes = LinearRegression()
    # scikit_linear_diabetes.fit(x_diabetes_train, y_diabetes_train)
    # print('Results on diabetes dataset using scikit-learn Linear Regression model')
    # # Test model on training set
    # scikit_predictions_diabetes_train = scikit_linear_diabetes.predict(x_diabetes_train)
    # scikit_scores_diabetes_train = mean_squared_error(
    #     scikit_predictions_diabetes_train, y_diabetes_train)
    # print('Training set mean accuracy: {:.4f}'.format(scikit_scores_diabetes_train))
    # # Test model on testing set
    # scikit_predictions_diabetes_test = scikit_linear_diabetes.predict(x_diabetes_test)
    # scikit_scores_diabetes_test = mean_squared_error(
    #     scikit_predictions_diabetes_test, y_diabetes_test)
    # print('Testing set mean accuracy: {:.4f}'.format(scikit_scores_diabetes_test))

    """ CUSTOM LOGISTIC
    Trains and tests our Logistic Regression model trained with Gradient Descent
    """
    # Trains our Logistic Regression model on Wisconsin cancer data
    # These inputs need to be "tuned"
    # Start by picking any one of theme

    # current best: train: 0.6152
    # current best:  test: 0.7047

    # Increased alpha to 0.04: 0.6171, 0.7568

    # t_cancer = 1000      # how long you want to run for
    # alpha_cancer = 1e-3   # how large of a step you take (started at 1e-4)
    # epsilon_cancer = 1e-8  # expected magnitude of movement at the finish line (1e-8)
    our_logistic_cancer = LogisticRegressionGradientDescent()

    log.info("LOGISTIC CANCER FIT ... ")

    format = "%10i %15g %15g %15g %15g"
    sformat = re.sub(r'[a-z]', 's', format)

    tc_list = [100, 200, 300]
    ac_list = [1e-4, 1e-3, 1e-2, 1e-1]
    ec_list = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    tc_list = [100]
    ac_list = [1e-4]
    ec_list = [1e-8]

    if args.t_cancer:
        tc_list = [args.t_cancer]
    if args.alpha_cancer:
        ac_list = [args.alpha_cancer]
    if args.epsilon_cancer:
        ec_list = [args.epsilon_cancer]

    for t_cancer in tc_list:
        print(sformat % (
            "t_cancer",
            "alpha_cancer",
            "epsilon_cancer",
            "cancer_train",
            "cancer_test"
        ))
        for alpha_cancer in ac_list:
            for epsilon_cancer in ec_list:

                our_logistic_cancer.fit(
                    x_cancer_train, y_cancer_train, t_cancer, alpha_cancer, epsilon_cancer)
                # print('Results on Wisconsin breast cancer dataset using our Logistic Regression model')
                # Test model on training set
                our_scores_cancer_train = our_logistic_cancer.score(x_cancer_train, y_cancer_train)
                # print('Training set mean accuracy: {:.4f}'.format(our_scores_cancer_train))
                # Test model on testing set
                our_scores_cancer_test = our_logistic_cancer.score(x_cancer_test, y_cancer_test)
                # print('Testing set mean accuracy: {:.4f}'.format(our_scores_cancer_test))

                print(format % (
                    t_cancer,
                    alpha_cancer,
                    epsilon_cancer,
                    our_scores_cancer_train,
                    our_scores_cancer_test
                ))

    # Trains our Logistic Regression model on digits 7 and 9 data
    # t_digits79 = 100
    # alpha_digits79 = 1e-4
    # epsilon_digits79 = 1e-8
    # our_logistic_digits79 = LogisticRegressionGradientDescent()

    # log.info("LOGISTIC DIGITS FIT ... ")
    # log.info("t_digits79: " + str(t_digits79))
    # log.info("alpha_digits79: " + str(alpha_digits79))
    # log.info("epsilon_digits79: " + str(epsilon_digits79))

    # our_logistic_digits79.fit(
    #     x_digits79_train, y_digits79_train, t_digits79, alpha_digits79, epsilon_digits79)
    # print('Results on digits 7 and 9 dataset using our Logistic Regression model')
    # # Test model on training set
    # our_scores_digits79_train = our_logistic_digits79.score(x_digits79_train, y_digits79_train)
    # print('Training set mean accuracy: {:.4f}'.format(our_scores_digits79_train))
    # # Test model on testing set
    # our_scores_digits79_test = our_logistic_digits79.score(x_digits79_test, y_digits79_test)
    # print('Testing set mean accuracy: {:.4f}'.format(our_scores_digits79_test))

#     """
#   Trains and tests our Linear Regression model trained using Gradient Descent
#   """
#     # Trains our Linear Regression model on Boston housing price data
#     t_housing = 0.0
#     alpha_housing = 0.0
#     epsilon_housing = 0.0
#     our_linear_housing = LinearRegressionGradientDescent()

#     log.info("LINEAR HOUSING FIT ... ")
#     our_linear_housing.fit(
#         x_housing_train, y_housing_train, t_housing, alpha_housing, epsilon_housing)
#     print('Results on Boston housing price dataset using our Linear Regression model')
#     # Test model on training set
#     our_predictions_housing_train = our_linear_housing.predict(x_housing_train)
#     our_scores_housing_train = mean_squared_error(our_predictions_housing_train, y_housing_train)
#     print('Training set mean accuracy: {:.4f}'.format(our_scores_housing_train))
#     # Test model on testing set
#     our_predictions_housing_test = our_linear_housing.predict(x_housing_test)
#     our_scores_housing_test = mean_squared_error(our_predictions_housing_test, y_housing_test)
#     print('Testing set mean accuracy: {:.4f}'.format(our_scores_housing_test))

#     # log.info("LINEAR HOUSING PREDICT ... ")
#     # predictions = our_linear_housing.predict(x_housing_test)

#     # Trains our Linear Regression model on diabetes data
#     t_diabetes = 0.0
#     alpha_diabetes = 0.0
#     epsilon_diabetes = 0.0
#     our_linear_diabetes = LinearRegressionGradientDescent()
#     log.info("LINEAR DIABETES FIT ... ")
#     our_linear_diabetes.fit(
#         x_diabetes_train, y_diabetes_train, t_diabetes, alpha_diabetes, epsilon_diabetes)
#     print('Results on diabetes dataset using our Linear Regression model')
#     # Test model on training set
#     our_predictions_diabetes_train = our_linear_diabetes.predict(x_diabetes_train)
#     our_scores_diabetes_train = mean_squared_error(our_predictions_diabetes_train, y_diabetes_train)
#     print('Training set mean accuracy: {:.4f}'.format(our_scores_diabetes_train))
#     # Test model on testing set
#     our_predictions_diabetes_test = our_linear_diabetes.predict(x_diabetes_test)
#     our_scores_diabetes_test = mean_squared_error(our_predictions_diabetes_test, y_diabetes_test)
#     print('Testing set mean accuracy: {:.4f}'.format(our_scores_diabetes_test))

#     # log.info("LINEAR DIABETES PREDICT ... ")
#     # predictions = our_linear_diabetes.predict(x_diabetes_train)
