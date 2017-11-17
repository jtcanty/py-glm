# A python implementation of GLMs #

import numpy as np

class GLM(object):
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

    def __init__(self, distribution='gaussian', n_iter=1000, conv_method='b-gradient', reg='l2', learning_rate=1,
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
            
        reg : str
            Name of the regularization term. Choices:
            l2 | l1 | elastic 

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
        self.reg = reg
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
        gradient = np.zeros(n)
        
        if self.conv_method == 'b-gradient':
            for i in range(0, n):
                if i == 0:
                    loss[i] = (1/m) * sum((np.dot(X, weight) - y) * X[:,i])
                elif i != 0:
                    loss[i] = (1/m) * sum((np.dot(X, weight) - y) * X[:,i]) + (self.lambd/m) * weight[i]
        
        elif self.conv_method == 's-gradient':
            if m == 0:
                loss = (np.dot(X, weight) - y) * X[n]
            elif m != 0:
                loss = (np.dot(X, weight) - y) * X[n] + (self.lambd) * weight[n] 
                
        elif self.conv_method == 'newton':      
            for i in range(0, n):
                gradient[i] = (1/m) * sum((np.dot(X, weight) - y) * X[:,i])
            inv_hessian = self.inverse_hessian(X, y, weight, m, n)
            loss = np.dot(inv_hessian, gradient)

 
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
        weight : array-like, shape = [n features]
            Vector of optimized weights
            
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
            for i in range(0, m):
                for j in range(0, n):
                    loss = self.loss(X[i,:], y[i], weight, i, j)
                    weight = weight - self.learning_rate * loss
        
        elif self.conv_method == 'newton':
            for i in range(0, self.n_iter):
                loss = self.loss(X, y, weight, m, n)
                weight = weight - self.learning_rate * loss

                
        return weight

    def predict(self, X, weight):
        '''Predicts response variables using the optimal weights

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Vector of training data
        weight : array-like, shape = [n features]
            Vector of optimized weights  

        Returns
        -------
        prediction : array-like, shape = [n_samples]
            Vector of predicted response
        '''
        
        prediction = np.dot(X, weight)
        
        return prediction
    
    def inverse_hessian(self, X, y, weight, m, n):
        '''Computes the Inverse of the Hessian of the log-likelihood function

          

        '''

        hessian = np.dot(np.transpose(X), X)    
        inv_hessian = np.linalg.inv(hessian)

        return inv_hessian
    
    def reg(self, ):
        '''Computes the regularization term for a training set
            
        Parameters
        ----------
        reg : str
            Name of the regularization term. Choices:
            l2 | l1 | elastic 

        Returns
        -------
            
        '''
        
        if self.reg == 'l1':
            
            
        elif self.reg == 'l2':
            
            
        elif self.reg == 'elastic':
            
        
    def mu(self):
        '''Computes the response function for different distributions
            
        Parameters
        ----------
        distribution : str
            Name of the distribution. Choices:
            gaussian | poisson | inv_gaussian | gamma | multinomial

        Returns
        -------
            
        '''
        

    