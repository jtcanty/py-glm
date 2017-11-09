# A python implementation of GLMs #

import numpy as np

class GLM:
    '''This class defines a generalized linear-model

    Parameters
    ----------
    learning_rate : integer
        Learning rate for computing the updated weights

    Attributes
    ----------
    weight : array-like, shape = [n_features]
        Vector of weights

    '''

    def __init__(self, distribution='gaussian', n_iter=1000, conv_method='b-gradient', learning_rate=.0001,
                    lambd=1):
        '''

        Parameters
        ----------
        distribution : str
            Name of the distribution. Choices:
            gaussian | poisson | inv_gaussian | gamma | multinomial

        n_iter : integer
            Number of iterations of to compute cost function

        conv_method : str
            Fitting method used:
            b-gradient | s-gradient | newton

        learning_rate : float
            Learning rate parameter implemented during gradient descent

        reg_param : float
            Regularization parameter

        Returns
        -------
        self : class instance
            Instance of the GLM class

        '''

        self.distribution = distribution
        self.n_iter = n_iter
        self.conv_method = conv_method
        self.learning_rate = learning_rate
        self.lambd = lambd

    def loss(self, X, y, weight, m, n):
        '''Computes the l2 regularized loss function

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Vector of training data
        y : array-like, shape = [n_samples]
            Vector of response data
        weight : array-like, shape = [n features]
            Vector of weights
        m : int
            Number of training examples
        n : int
            Number of features (including intercept)

        Returns
        -------
        loss : int
            Computed loss 

        '''
        loss = np.zeros(n)
        
        if self.conv_method == 'b-gradient':
            for i in range(0,n):
                if i == 0:
                    loss[i] = (1/m) * sum((np.dot(X, weight) - y) * X[:,i])
                elif i != 0:
                    loss[i] = (1/m) * sum((np.dot(X, weight) - y) * X[:,i]) + (self.lambd/m) * weight[i]
        
        elif self.conv_method == 's-gradient':
            for i in range(0, n):
                if i == 0:
                        
            
        #elif self.conv_method == 'newton':
             
        return loss
    
    def fit(self, X, y):
        '''Computes the optimal weights that minimize the cost function

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Vector of training data
        y : array-like, shape = [n_samples]
            Vector of response data
        weight : array-like, shape = [n features]
            Vector of weights
        m : int
            Number of training examples
        n : int
            Number of features (including intercept)

        Returns
        -------

        '''
        
        
        m = np.shape(X)[0]
        intercept = np.ones(m)
        X = np.vstack((np.transpose(intercept), X))
        X = np.transpose(X)
        
        n = np.shape(X)[1]
        weight = np.zeros(n)

        if self.conv_method == 'b-gradient':
            for i in range(0, self.n_iter):
                loss = self.loss(X, y, weight, m, n)
                weight = weight - self.learning_rate * loss


        elif self.conv_method == 's-gradient':
            for i in range(0, self.n_iter)
                for i in range(0, m):
                    loss = self.loss(X, y, weight, m, n)
                    weight = weight - self.learning_rate * loss

        #elif self.conv_method == 'newton':


        return weight

    def predict(self, X):
        '''Predicts response variables using the optimal weights

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Vector of training data

        Returns
        -------

        '''

        return 'b'