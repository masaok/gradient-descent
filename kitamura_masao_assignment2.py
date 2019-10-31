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
Name: Kitamura, Masao

Collaborators: Huerta, Emilia

Collaboration details: Discussed all implementation details with Jane Doe.

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
Training set mean accuracy: 0.914062
Testing set mean accuracy: 0.947368
Results on digits 7 and 9 dataset using our Logistic Regression model
Training set mean accuracy: 1.0000
Testing set mean accuracy: 1.0000
Results on Boston housing price dataset using our Linear Regression model
Training set mean accuracy: 51.5090
Testing set mean accuracy: 42.8980
Results on diabetes dataset using our Linear Regression model
Training set mean accuracy: 2991.9850
Testing set mean accuracy: 1735.9390
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

            # Gradients start at zeros with same shape as x
            gradients = np.zeros(x.shape)

            # For every row in x ...
            for n in range(x.shape[0]):

                # Get row n of x
                x_n = x[n, ...]

                # Predictions
                h_x = np.dot(np.squeeze(w), x_n)  # scalar

                # Gradient Descent, slide 22
                gradients[n, :] = (-y[n] * x_n) / (1.0 + np.exp(y[n] * h_x))

            # Return the mean of the first column of gradients
            return np.mean(gradients, axis=0)  # How does this match the slide 22?

        elif loss_func == 'mean_squared':

            # Gradients start at zeros with same shape as x
            gradients = np.zeros(x.shape)

            # For every row in x ...
            for n in range(x.shape[0]):

                # Get a row of x at n
                x_n = x[n, :]

                # Take the dot product (wTx)
                h_x_n = np.dot(np.squeeze(w), x_n)

                # Compute gradients
                gradients[n] = (h_x_n - y[n]) * x_n

            # Return twice the mean
            return 2 * np.mean(gradients, axis=0)

        elif loss_func == 'half_mean_squared':

            # Gradients start at zeros with same shape as x
            gradients = np.zeros(x.shape)

            # For every row in x ...
            for n in range(x.shape[0]):

                # Get a row of x at n
                x_n = x[n, :]

                # Take the dot product (wTx)
                h_x_n = np.dot(np.squeeze(w), x_n)

                # Compute gradients
                gradients[n] = (h_x_n - y[n]) * x_n

            # Return the mean
            return np.mean(gradients, axis=0)

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

        # Convert all zero y's to -1, because predictions will only be 1 or -1
        y = np.where(y == 0.0, -1.0, 1.0)

        # +1 is the threshold
        self.__weights = np.zeros([1, x.shape[1]+1])
        self.__weights[0] = -1

        for i in range(int(t)):
            # log.info("LOGISTIC CANCER FIT LOOP i=" + str(i))

            # predict
            h_x = self.predict(x)

            # compute the loss (Gradient Descent, slide 21)
            loss = np.mean(np.log(1 + np.exp(-y * h_x)))  # (N, 1) and (N, 1)

            # call update (which calls compute the gradients)
            w_i = self.__optimizer.update(self.__weights, x, y, alpha, 'logistic')

            if loss == 0:  # global minima
                log.info("LOSS is ZERO; we're done!")
                break

            # d_w is the change in weights
            d_w = self.__weights - w_i

            # magnitude of the change in weights
            mag = np.sqrt(np.sum(d_w ** 2))

            if i % 100 == 0:
                log.info("i=" + str(i) + " loss=" + str(loss) + " mag=" + str(mag))

            # Save the new weights
            self.__weights = w_i

            # check stopping conditions
            if mag < epsilon:  # epsilon is expected mag when done
                log.info("mag: " + str(mag))
                log.info("eps: " + str(epsilon))
                log.info("MAG < EPSILON; we're done!")
                break

    def predict(self, x):
        """
        Predicts the label for each feature vector x

        x : N x d feature vector

        returns : N x 1 label vector
        """

        # Weights w is shape [1, d + 1] (number of dimensions)
        # The extra weight is -1 * threshold

        # Prepend the set of x attributes with 0.5 bias (in Perceptron, this is threshold)
        bias = 0.5 * np.ones([x.shape[0], 1])

        # Concatenate the bias above
        x = np.concatenate([bias, x], axis=-1)  # x is now [N, d + 1]

        # Prepare an array for predictions
        predictions = np.zeros(x.shape[0])

        # Iterate through every row (example) in x
        for n in range(x.shape[0]):
            x_n = x[n, ...]   # ... and : are the same

            # With dot product, the size of the middle elements must match
            h_x = np.dot(np.squeeze(self.__weights.T), x_n)

            # Take the exponent of the negative predictions
            exp = np.exp(-1 * h_x)

            predictions[n] = 1 / (1 + exp)  # Logistic Reg, slide 11

        # Using a threshold of 0.5, generate the predictions
        predictions = np.where(predictions >= 0.5, 1.0, -1.0)

        return predictions

    def score(self, x, y):
        """
        Predicts labels based on feature vector x and computes the mean accuracy
        of the predictions

        x : N x d feature vector
        y : N x 1 ground-truth label

        returns : double
        """
        predictions = self.predict(x)

        # Convert all zero answers to -1 to prepare matching with predictions
        y = np.where(y == 0.0, -1.0, 1.0)

        # Scores are based on where predictions match y
        scores = np.where(y == predictions, 1, 0)

        # Return the average
        return np.mean(scores)


"""
Implementation of our Linear Regression model trained using Gradient Descent
"""


class LinearRegressionGradientDescent(object):
    def __init__(self):
        # Define private variables
        self.__weights = None
        self.__optimizer = GradientDescentOptimizer()

    def fit(self, x, y, t, alpha, epsilon):  # Linear
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

        for i in range(t):

            # predict
            h_x = self.predict(x)

            # compute the loss
            loss = np.mean((h_x - y) ** 2)

            # compute the gradients
            w_i = self.__optimizer.update(self.__weights, x, y, alpha, 'mean_squared')

            # check stopping conditions
            if loss == 0:  # global minima (we've predicted perfectly)
                break

            # Calculate change in weights
            d_w = self.__weights - w_i

            # Magnitude of the change in weights (sqrt of sum of squares)
            mag = np.sqrt(np.sum(d_w ** 2))

            # Debug output
            if i % 100 == 0:
                log.info("i=" + str(i) + " loss=" + str(loss) + " mag=" + str(mag))

            # Save the new weights
            self.__weights = w_i

            # check stopping conditions
            if mag < epsilon:  # epsilon is expected mag when done
                log.info("mag: " + str(mag))
                log.info("eps: " + str(epsilon))
                break

    def predict(self, x):  # Linear
        """
        Predicts the label for each feature vector x

        x : N x d feature vector

        returns : N x 1 label vector
        """

        # what should weights be equal to?
        x = np.concatenate([0.5 * np.ones([x.shape[0], 1]), x], axis=-1)

        # Predictions start at zeros
        h_x = np.zeros(x.shape[0])

        # For every row in x
        for n in range(x.shape[0]):

            # Get the row
            x_n = x[n, ...]

            # Expand a single dimension to allow the dot product to work
            x_n = np.expand_dims(x_n, axis=-1)

            # Take the dot product (scalar)
            h_x_n = np.dot(np.squeeze(self.__weights), x_n)

            # Assign value to array of predictions
            h_x[n] = h_x_n

        # Return predictions
        return h_x

    def score(self, x, y):
        """
        Predicts labels based on feature vector x and computes the
        mean squared loss of the predictions

        x : N x d feature vector
        y : N x 1 ground-truth label

        returns : double
        """
        # Take the exam
        h_x = self.predict(x)  # (N x 1)

        # Grade the exam
        return np.mean(h_x - y) ** 2


def mean_squared_error(y_hat, y):  # Gradient Descent, slide 15
    """
    Computes the mean squared error

    y_hat : N x 1 predictions
    y : N x 1 ground-truth label

    returns : double
    """
    y_diff = y_hat - y  # vector subtraction

    # Return the average of the differences squared
    return np.mean(y_diff ** 2)


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

    # Run everything by default
    logistic_cancer = True
    logistic_cancer_verbose = 0
    logistic_digits = True
    linear_housing = True
    linear_diabetes = True

    # Disable parts of the assignment by uncommenting lines below
    # logistic_cancer = False
    # logistic_digits = False
    # linear_housing = False
    # linear_diabetes = False

    if logistic_cancer:

        # Trains our Logistic Regression model on Wisconsin cancer data

        # t_cancer = 1000        # how long you want to run for
        # alpha_cancer = 1e-3    # how large of a step you take (started at 1e-4)
        # epsilon_cancer = 1e-8  # expected magnitude of movement at the finish line (1e-8)
        our_logistic_cancer = LogisticRegressionGradientDescent()

        log.info("LOGISTIC CANCER FIT ... ")

        format = "%10i %15g %15g %15g %15g"  # numeric data in row format
        sformat = re.sub(r'[a-z]', 's', format)  # use same widths, but format for strings only

        tc_list = [100, 200, 300]
        ac_list = [1e-4, 1e-3, 1e-2, 1e-1]
        ec_list = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

        tc_list = [1000]
        # tc_list = [3]
        ac_list = [10]
        ec_list = [1e-11]

        for t_cancer in tc_list:
            for alpha_cancer in ac_list:
                for epsilon_cancer in ec_list:

                    log.info("t: " + str(t_cancer))
                    log.info("alpha: " + str(alpha_cancer))
                    log.info("epsilon: " + str(epsilon_cancer))

                    our_logistic_cancer.fit(
                        x_cancer_train, y_cancer_train, t_cancer, alpha_cancer, epsilon_cancer)
                    # print('Results on Wisconsin breast cancer dataset using our Logistic Regression model')

                    # Test model on training set
                    our_scores_cancer_train = our_logistic_cancer.score(
                        x_cancer_train, y_cancer_train)
                    # print('Training set mean accuracy: {:.4f}'.format(our_scores_cancer_train))

                    # Test model on testing set
                    our_scores_cancer_test = our_logistic_cancer.score(x_cancer_test, y_cancer_test)
                    # print('Testing set mean accuracy: {:.4f}'.format(our_scores_cancer_test))

                    print(sformat % (
                        "t_cancer",
                        "alpha_cancer",
                        "epsilon_cancer",
                        "cancer_train",
                        "cancer_test"
                    ))
                    print(format % (
                        t_cancer,
                        alpha_cancer,
                        epsilon_cancer,
                        our_scores_cancer_train,
                        our_scores_cancer_test
                    ))

        # Trains scikit-learn Logistic Regression model on Wisconsin cancer data
        scikit_logistic_cancer = LogisticRegression(
            solver='liblinear', verbose=logistic_cancer_verbose)
        scikit_logistic_cancer.fit(x_cancer_train, y_cancer_train)
        print('Results on Wisconsin breast cancer dataset using scikit-learn Logistic Regression model')
        # Test model on training set
        scikit_scores_cancer_train = scikit_logistic_cancer.score(x_cancer_train, y_cancer_train)
        print('Training set mean accuracy: {:.4f}'.format(scikit_scores_cancer_train))
        # Test model on testing set
        scikit_scores_cancer_test = scikit_logistic_cancer.score(x_cancer_test, y_cancer_test)
        print('Testing set mean accuracy: {:.4f}'.format(scikit_scores_cancer_test))

    if logistic_digits:

        # Trains our Logistic Regression model on digits 7 and 9 data

        # This tuning results in loss decreasing to a constant value
        # This means that the training set tells us everything there is to know
        # about the testing set.
        # "We've learned all we need to, to do well on the testing set."
        # Exam analogy: if you are studying for an exam,
        #   ...and you don't know everything in the study guide...
        #   ...but, you happened to ace the test with what you do know...
        #   ...this is what happens with these inputs.
        t_digits79 = 1000
        alpha_digits79 = 0.001
        epsilon_digits79 = 1e-8
        our_logistic_digits79 = LogisticRegressionGradientDescent()

        log.info("LOGISTIC DIGITS FIT ... ")
        log.info("t_digits79: " + str(t_digits79))
        log.info("alpha_digits79: " + str(alpha_digits79))
        log.info("epsilon_digits79: " + str(epsilon_digits79))

        our_logistic_digits79.fit(
            x_digits79_train, y_digits79_train, t_digits79, alpha_digits79, epsilon_digits79)
        print('Results on digits 7 and 9 dataset using our Logistic Regression model')
        # Test model on training set
        our_scores_digits79_train = our_logistic_digits79.score(x_digits79_train, y_digits79_train)
        print('Training set mean accuracy: {:.4f}'.format(our_scores_digits79_train))
        # Test model on testing set
        our_scores_digits79_test = our_logistic_digits79.score(x_digits79_test, y_digits79_test)
        print('Testing set mean accuracy: {:.4f}'.format(our_scores_digits79_test))

        # """ SCI KIT LEARN LOGISTIC DIGITS
        # Trains and tests Logistic Regression model from scikit-learn
        # """

        # Trains scikit-learn Logistic Regression model on digits 7 and 9 data
        scikit_logistic_digits79 = LogisticRegression(solver='liblinear')
        scikit_logistic_digits79.fit(x_digits79_train, y_digits79_train)
        print('Results on digits 7 and 9 dataset using scikit-learn Logistic Regression model')
        # Test model on training set
        scikit_scores_digits79_train = scikit_logistic_digits79.score(
            x_digits79_train, y_digits79_train)
        print('Training set mean accuracy: {:.4f}'.format(scikit_scores_digits79_train))
        # Test model on testing set
        scikit_scores_digits79_test = scikit_logistic_digits79.score(
            x_digits79_test, y_digits79_test)
        print('Testing set mean accuracy: {:.4f}'.format(scikit_scores_digits79_test))

    """
    Trains and tests our Linear Regression model trained using Gradient Descent
    """

    if linear_housing:

        # Trains our Linear Regression model on Boston housing price data
        t_housing = 30000
        alpha_housing = 3.25e-6
        epsilon_housing = 1e-8
        our_linear_housing = LinearRegressionGradientDescent()

        log.info("LINEAR HOUSING FIT ... ")
        our_linear_housing.fit(
            x_housing_train, y_housing_train, t_housing, alpha_housing, epsilon_housing)
        print('Results on Boston housing price dataset using our Linear Regression model')

        # Test model on training set
        our_predictions_housing_train = our_linear_housing.predict(x_housing_train)
        our_scores_housing_train = mean_squared_error(
            our_predictions_housing_train, y_housing_train)
        print('Training set mean accuracy: {:.4f}'.format(our_scores_housing_train))

        # Test model on testing set
        our_predictions_housing_test = our_linear_housing.predict(x_housing_test)
        our_scores_housing_test = mean_squared_error(our_predictions_housing_test, y_housing_test)
        print('Testing set mean accuracy: {:.4f}'.format(our_scores_housing_test))

        # """ SCI KIT LEARN LINEAR HOUSING
        # Trains and tests Linear Regression model from scikit-learn
        # """
        # Trains scikit-learn Linear Regression model on Boston housing price data
        scikit_linear_housing = LinearRegression()
        scikit_linear_housing.fit(x_housing_train, y_housing_train)
        print('Results on Boston housing price dataset using scikit-learn Linear Regression model')
        # Test model on training set
        scikit_predictions_housing_train = scikit_linear_housing.predict(x_housing_train)
        scikit_scores_housing_train = mean_squared_error(
            scikit_predictions_housing_train, y_housing_train)
        print('Training set mean accuracy: {:.4f}'.format(scikit_scores_housing_train))
        # Test model on testing set
        scikit_predictions_housing_test = scikit_linear_housing.predict(x_housing_test)
        scikit_scores_housing_test = mean_squared_error(
            scikit_predictions_housing_test, y_housing_test)
        print('Testing set mean accuracy: {:.4f}'.format(scikit_scores_housing_test))

    if linear_diabetes:

        # DIABETES LINEAR
        # Trains our Linear Regression model on diabetes data

        # Iterations (how long to run for)
        # 10,000  = 2994.6357
        # 100,000 = 2991.9850 (train) and 1735.9390 (test)
        t_diabetes = 100000

        # Learning rate (how large of a step)
        # If loss goes infinite, alpha is too large
        # If loss decreases but too slowly, alpha is too small
        # Use binary search to tune this
        alpha_diabetes = 2.5

        # Magnitude of change in weights at the finish line
        # Epsilon does not matter here because ... (I forgot the reason) ???
        epsilon_diabetes = 1e-8

        our_linear_diabetes = LinearRegressionGradientDescent()
        log.info("LINEAR DIABETES FIT ... ")
        our_linear_diabetes.fit(
            x_diabetes_train, y_diabetes_train, t_diabetes, alpha_diabetes, epsilon_diabetes)
        print('Results on diabetes dataset using our Linear Regression model')
        # Test model on training set
        our_predictions_diabetes_train = our_linear_diabetes.predict(x_diabetes_train)
        our_scores_diabetes_train = mean_squared_error(
            our_predictions_diabetes_train, y_diabetes_train)
        print('Training set mean accuracy: {:.4f}'.format(our_scores_diabetes_train))
        # Test model on testing set
        our_predictions_diabetes_test = our_linear_diabetes.predict(x_diabetes_test)
        our_scores_diabetes_test = mean_squared_error(
            our_predictions_diabetes_test, y_diabetes_test)
        print('Testing set mean accuracy: {:.4f}'.format(our_scores_diabetes_test))

        # SCIKIT DIABETES LINEAR
        # Trains scikit-learn Linear Regression model on diabetes data
        log.info("SCI KIT LEARN LINEAR DIABETES")
        scikit_linear_diabetes = LinearRegression()
        scikit_linear_diabetes.fit(x_diabetes_train, y_diabetes_train)
        print('Results on diabetes dataset using scikit-learn Linear Regression model')
        # Test model on training set
        scikit_predictions_diabetes_train = scikit_linear_diabetes.predict(x_diabetes_train)
        scikit_scores_diabetes_train = mean_squared_error(
            scikit_predictions_diabetes_train, y_diabetes_train)
        print('Training set mean accuracy: {:.4f}'.format(scikit_scores_diabetes_train))
        # Test model on testing set
        scikit_predictions_diabetes_test = scikit_linear_diabetes.predict(x_diabetes_test)
        scikit_scores_diabetes_test = mean_squared_error(
            scikit_predictions_diabetes_test, y_diabetes_test)
        print('Testing set mean accuracy: {:.4f}'.format(scikit_scores_diabetes_test))
