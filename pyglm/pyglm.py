# A python implementation of GLMs #

import numpy as np
import random

DISTR_TYPES = {'gaussian': 0, 'poisson': 1, 'inv_gaussian': 2, 'exponential': 3, 'categorical': 4}
LEARNING_METHOD = {'BGD': 0, 'SGD': 1, 'N': 2}
REGULARIZATION_METHOD = {'l1': 0 ,'l2': 1, 'elasticnet': 2}
TASK = {'classification': 0, 'regression': 1}

def _mu(distr, x, beta):
    '''Computes the response function for different distributions

    Parameters
    ----------
    dist : str
        Name of the distribution. Choices:
        gaussian | poisson | inv_gaussian | exponential | categorical
        
    beta : array-like, shape = [n features]
                Vector of weights
    x : array-like, shape = [n_samples, n_features]
            Vector of training data
    
    Returns
    -------
    mu : float
        Expectation value of the distribution function for a single sample
        '''
    if distr == 'gaussian':
        mu = np.dot(x, beta)
        return mu

    elif distr == 'poisson':
        nu = np.dot(x, beta)
        mu = np.exp(nu)
        return mu

    elif distr == 'inv_gaussian':
        nu = np.dot(x, beta)
        mu = np.sqrt(nu)
        return mu

    elif distr == 'exponential':
        nu = np.dot(x, beta)
        mu = 1 / nu
        return mu

    elif distr == 'categorical':
        nu = np.dot(x, beta)
        mu = 1 / (1 + np.exp(-nu))
        return mu

def _inverse_hessian(x):
    '''Computes the Inverse of the Hessian of the log-likelihood function

    x : array-like, shape = [n_samples, n_features]
        Vector of training data

    Returns
    -------
    inv_hessian : array-like, shape = [n_features, n_features]
       Inverse of the hessian for the loss function
    '''
    hessian = np.dot(np.transpose(x), x)    
    inv_hessian = np.linalg.inv(hessian)

    return inv_hessian


def _reg(lambd, alpha, beta):
    '''Computes the regularization term for a training set

    Parameters
    ----------   
    lambd : float
            General regularization parameter
            
    alpha : float
        Elastic net regularization parameter
        alpha = 0
        
    beta : array-like, shape = [n features]
                Vector of weights

    Returns
    -------
    reg : float
        Value computed from regularization   
    '''
    reg = lambd * alpha * beta + lambd * (1 - alpha)
    
    return reg


def _bgd(n_iter, distr, descent, 
         lambd, alpha, learning_rate, 
         x, y, beta, 
         n_samples, n_features):          
    '''Implements variable-size batch gradient descent method'''
    for i in range(0, n_iter):  
        
        # Initialize array of gradients for each feature
        grad = np.zeros(n_features)
        
        # Compute gradient for each feature
        for j in range(0, n_features):
            mu = _mu(distr, x, beta)
            grad_update = sum(_grad(distr, mu, y) * x[:,j])
            
            # Apply regularization
            if j == 0:
                grad[j] = (1 / n_samples) * grad_update    
            elif j != 0:
                grad[j] = (1 / (2 * n_samples)) * grad_update + _reg(lambd, alpha, beta[j])
        
        beta -= learning_rate * grad
     
    return beta
    
    
def _sgd(n_iter, distr, descent, 
         lambd, alpha, learning_rate, 
         x, y, beta, 
         n_samples, n_features):
    '''Implements stochastic gradient descent method'''
    for i in range(0, n_samples):
        
        # Initialize array of gradients for each feature
        grad = np.zeros(n_features)
        
        # Compute gradient for each feature
        for j in range(0, n_features):       
            mu = _mu(distr, x, beta)
            grad_update = _grad(distr, mu[i], y[i]) * x[i,j]
            
            # Apply regularization
            if j == 0:
                grad = grad_update
            elif j != 0:
                grad = grad_update + _reg(lambd, alpha, beta[j])
            
        beta -= learning_rate * grad
            
    return beta


def _newton(n_iter, distr, descent, 
            lambd, alpha, learning_rate, 
            x, y, beta, 
            n_samples, n_features):
    '''Implements newton coordinate descent method''' 
    for i in range(0, n_iter):
        
        # Initialize array of gradients for each feature
        grad = np.zeros(n_features)
        
        # Compute gradients for each feature
        for j in range(0, n_features):
            mu = _mu(distr, x, beta)     
            grad_update = sum(_grad(distr, mu, y) * x[:,j])
            grad[j] = (1 / n_samples) * grad_update

        # Compute inverse Hessian matrix
        inv_hessian = _inverse_hessian(x)
        loss = np.dot(inv_hessian, grad)
            
        beta -= learning_rate * loss
        
    return beta

    
def _grad(distr, mu, y):
    '''Computes the gradient of empirical loss function. 

    Parameters
    ----------
    distr : str
        Name of the distribution. Choices:
        gaussian | poisson | inv_gaussian | exponential | multinomial
        
    mu : float, array-like
        Expectation value of the distribution function for a single sample
    
    x : float, array-like

    y : float, array-like, shape = [n_samples]
        Vector of response data

    Returns
    -------
    loss : int
        Computed loss 
    '''
    grad = mu - y
        
    return grad

#def _cross_validation():
    
class GLM(object):
    '''This class defines an object for estimating a generalized linear-model

    Parameters
    ----------
    learning_rate : integer
        Learning rate for computing the updated weights

    Attributes
    ----------
    beta : array-like, shape = [n_features]
        Vector of weights
    '''

    def __init__(self, 
                 distr='gaussian', 
                 task='regression',
                 n_iter=100, 
                 descent='BGD', 
                 regularization='elasticnet',
                 learning_rate=1, 
                 lambd=1, 
                 alpha=0.5):
        '''

        Attributes
        ----------
        distr : str
            Name of the distribution. Choices:
            gaussian | poisson | inv_gaussian | exponential | multinomial
            
        task: str
            Learning task.
            regression | classification

        n_iter : integer
            Number of iterations of to compute cost function

        descent : str
            Fitting method used:
            BGD | SGD | N
            
        regularization : str
            Name of regularization method used:
            l1 | l2 | elasticnet

        learning_rate : float
            Learning rate parameter implemented during gradient descent

        lambd : float
            General regularization parameter
            
        alpha : float
            Elastic net regularization parameter
            alpha = 0
      

        Returns
        -------
        self : class instance
            Instance of the GLM class
        '''

        self.distr = distr
        self.task = task
        self.n_iter = n_iter
        self.descent = descent
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.lambd = lambd
        self.alpha = alpha

    def check_parameters(self):
        '''Check the input parameters'''
        if self.n_iter <= 0:
            raise ValueError('n_iter must be a positive integer')
        
    def get_task(self):
        '''Checks the whether the task is classification or regularization'''
        try:
            return TASK[self.task]
        except KeyError:
            raise ValueError('Task %s is not supported'
                             % self.task)
            
    def get_distribution(self):
        '''Checks the error distribution to be used'''
        try:
            return DISTR_TYPES[self.distr]
        except KeyError:
            raise ValueError('Distribution %s is not supported'
                             % self.distr)
        
    def get_learning_method(self):
        '''Validates the learning method'''
        try:
            return LEARNING_METHOD[self.descent]
        except KeyError:
            raise ValueError('Learning method %s is not supported'
                             % self.descent)
        
    def get_regularization_type(self):
        '''Validates the regularization method'''
        try:
            return REGULARIZATION_METHOD[self.regularization]
        except KeyError:
            raise ValueError('Regularization method %s is not supported'
                             % self.regularization)
        
    def fit(self, x, y):
        '''Computes the optimal weights that minimize the cost function

        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Vector of training data
        y : array-like, shape = [n_samples]
            Vector of response data

        Returns
        -------
        beta : array-like, shape = [n features]
            Vector of optimized weights   
        '''
        
        # Check constants and parameters
        self.check_parameters()
        self.get_task()
        self.get_distribution()
        self.get_learning_method()
        self.get_regularization_type()
        
        n_samples = np.shape(x)[0]
        if n_samples != np.shape(y)[0]:
            raise ValueError('Dimension of response should be (n_samples,)')
            
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        if not isinstance(y, np.ndarray):
            y = np.array(y)
                   
        # Construct bias term
        bias = np.ones(n_samples)
        x = np.vstack((bias, x))
        x = x.T

        n_features = np.shape(x)[1]
        beta = np.ones(n_features)
        
        # Select descent method
        if self.descent == 'BGD':
            beta = _bgd(self.n_iter, self.distr, self.descent, 
                        self.lambd, self.alpha, self.learning_rate,
                        x, y, beta, 
                        n_samples, n_features)
            
        elif self.descent == 'SGD':
            beta = _sgd(self.n_iter, self.distr, self.descent, 
                        self.lambd, self.alpha, self.learning_rate,
                        x, y, beta, 
                        n_samples, n_features)
            
        elif self.descent == 'N':
            beta = _newton(self.n_iter, self.distr, self.descent, 
                           self.lambd, self.alpha, self.learning_rate,
                           x, y, beta, 
                           n_samples, n_features)
                
        return beta

    
    def predict(self, x, beta):
        '''Predicts response variables using the optimal weights

            Parameters
            ----------
            x : array-like, shape = [n_samples, n_features]
                Vector of training data

            beta : array-like, shape = [n features]
                Vector of optimized weights  

            Returns
            -------
            yhat : array-like, shape = [n_samples]
                Vector of predicted response
        '''
        if not isinstance(x, np.ndarray):
            raise ValueError('Training data should be of type %s'
                             % np.ndarray)

        yhat = np.dot(x, beta) 
        return yhat    