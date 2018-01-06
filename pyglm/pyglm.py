# A python implementation of GLMs #

import numpy as np
import random

DISTRIBUTION = {'gaussian': 0, 'poisson': 1, 'inv_gaussian': 2, 'exponential': 3, 'categorical': 4, 'softmax': 5}
DESCENT = {'bgd': 0, 'newton': 1}
PENALTY = {'l1': 0 ,'l2': 1, 'elasticnet': 2}
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


def _penalty(lambd, alpha, beta):
    '''Computes the penalty term

    Parameters
    ----------   
    lambd : float
            General regularization parameter
            
    alpha : float
        Elastic net regularization parameter
        alpha = 0
        
    beta : array-like, shape = (1 + n features,)
                Vector of weights

    Returns
    -------
    reg : float
        Value computed from regularization   
    '''
    P = lambd * alpha * beta + lambd * (1 - alpha)
    
    return P


def _bgd(distr, descent, 
         lambd, alpha, lrn_rate, 
         x, y, beta):          
    '''Implements one cycle of variable-size batch gradient descent 
    
    Parameters
    ----------    
    distr : str
        Name of the distribution. Choices:
        gaussian | poisson | inv_gaussian | exponential | multinomial
    
    descent : str
        Fitting method used:
        bgd | newton
    
    lambd : float
        General regularization parameter

    alpha : float
        Elastic net regularization parameter
    
    lrn_rate : float
        Learning rate parameter implemented during gradient descent
        
    x : float, array-like
        Training data
        
    y : float, array-like, shape = [n_samples]
        Vector of response data
 
    beta : array-like, shape = (1 + n features,)
        Vector of weights 
        
    Returns
    -------
    beta : array-like, shape = (1 + n_features,)
        Updated parameters
        '''  
    n_samples = x.shape[0]
    n_features = x.shape[1]

    # Initialize array of gradients for each feature
    grad = np.zeros(n_features)

    # Compute gradient for each feature
    for j in range(0, n_features):
        mu = _mu(distr, x, beta)
        grad_batch = sum(_grad(distr, mu, y) * x[:,j])

        # Apply penalty
        P = _penalty(lambd, alpha, beta[j])
        if j == 0:
            grad[j] = (1 / n_samples) * grad_batch   
        elif j != 0:
            grad[j] = (1 / (2 * n_samples)) * grad_batch + P
        
        beta -= lrn_rate * grad
     
    return beta

def _newton(distr, descent, 
            lambd, alpha, lrn_rate, 
            x, y, beta):
    '''Implements one cycle of newton coordinate descent method
    
    Parameters
    ----------
    distr : str
        Name of the distribution. Choices:
        gaussian | poisson | inv_gaussian | exponential | multinomial
    
    descent : str
        Fitting method used:
        bgd | newton
    
    lambd : float
        General regularization parameter

    alpha : float
        Elastic net regularization parameter
    
    lrn_rate : float
        Learning rate parameter implemented during gradient descent
        
    x : float, array-like
        Training data
        
    y : float, array-like, shape = [n_samples]
        Vector of response data
 
    beta : array-like, shape = (1 + n features,)
        Vector of weights
        
    Returns
    -------
    beta : array-like, shape = (1 + n_features,)
        Updated parameters
    '''    
    n_samples = x.shape[0]
    n_features = x.shape[1]
    
    # Initialize array of gradients for each feature
    grad = np.zeros(n_features)
        
    # Compute gradients for each feature
    for j in range(0, n_features):
        mu = _mu(distr, x, beta)     

        grad_batch = sum(_grad(distr, mu, y) * x[:,j])
        grad[j] = (1 / n_samples) * grad_batch

    # Compute inverse Hessian matrix
    inv_hessian = _inverse_hessian(x)
    loss = np.dot(inv_hessian, grad)

    beta -= lrn_rate * loss
        
    return beta


def _l2_loss(distr, beta, x, y, 
             lambd, alpha):
    '''Compute the L2 loss function'''
    n_samples = x.shape[0]
    mu = _mu(distr, x, beta)
    P = sum(_penalty(lambd, alpha, beta))
    loss = sum((mu - y)**2)
    L2_loss = (1 / n_samples) * loss + P
    
    return L2_loss


def _cross_entropy_loss(distr, beta, x, y, 
                        lambd, alpha):
    '''Compute the cross-entropy loss function'''
    n_samples = x.shape[0]
    mu = _mu(distr, x, beta)
    P = sum(_penalty(lambd, alpha, beta))
    loss = sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))
    CE_loss = 0.5 * n_samples * loss + P
    
    return CE_loss
    

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
    
    
class GLM(object):
    '''This class defines an object for estimating a generalized linear-model

    Parameters
    ----------
    lrn_rate : integer
        learning rate for computing the updated weights

    Attributes
    ----------
    beta : array-like, shape = [n_features]
        Vector of weights
    '''

    def __init__(self, 
                 distr='gaussian', 
                 task='regression',
                 n_iter=1000, 
                 descent='newton', 
                 penalty='elasticnet',
                 lrn_rate=1e-2, 
                 lambd=1, 
                 alpha=0.5,
                 batch_sz = None,
                 beta0_ = None,
                 beta_ = None,
                 tol = 1e-3):
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
            
        penalty : str
            Name of regularization method used:
            l1 | l2 | elasticnet

        lrn_rate : float
            Learning rate parameter implemented during gradient descent

        lambd : float
            General regularization parameter
            
        alpha : float
            Elastic net regularization parameter
            alpha = 0
            
        batch_sz : float
            Size of sample size batch for each training iteration
            
        beta0_ : float
            Intercept term for the linear predictor 
        
        beta_ : array-like, shape = (n_features, )
            Vector of parameters for the linear predictor
            
        tol : float
            Tolerance for descent convergence
      

        Returns
        -------
        self : class instance
            Instance of the GLM class
        '''

        self.distr = distr
        self.task = task
        self.n_iter = n_iter
        self.descent = descent
        self.penalty = penalty
        self.lrn_rate = lrn_rate
        self.lambd = lambd
        self.alpha = alpha
        self.batch_sz = batch_sz
        self.beta0_ = beta0_
        self.beta_ = beta_
        self.tol = tol

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
            return DISTRIBUTION[self.distr]
        except KeyError:
            raise ValueError('Distribution %s is not supported'
                             % self.distr)
        
    def get_descent(self):
        '''Validates the learning method'''
        try:
            return DESCENT[self.descent]
        except KeyError:
            raise ValueError('Learning method %s is not supported'
                             % self.descent)
        
    def get_penalty(self):
        '''Validates the regularization method'''
        try:
            return PENALTY[self.penalty]
        except KeyError:
            raise ValueError('Regularization method %s is not supported'
                             % self.penalty)
        
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
        self.get_descent()
        self.get_penalty()
        
        n_samples = x.shape[0]
        
        # Add intercept to training data
        intercept = np.ones(n_samples)
        x = np.column_stack((intercept, x))
        
        n_features = x.shape[1]
  
        if n_samples != y.shape[0]:
            raise ValueError('Dimension of response should be (n_samples,)')
            
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        # Initialize parameters
        beta = np.zeros(n_features)
        if self.beta0_ is None and self.beta_ is None:
            beta[0] = abs(np.random.normal(0, 0.001, 1))
            beta[1:] = abs(np.random.normal(0, 0.001, n_features-1))
            
        else:
            beta[0] = self.beta0_ 
            beta[1:] = self.beta_
        
        # Initialize descent and accumulate losses
        L = []
        
        for i in range(0, self.n_iter):   
            if self.descent == 'bgd':
                beta = _bgd(self.distr, self.descent, 
                           self.lambd, self.alpha, self.lrn_rate, 
                           x, y, beta)  
                
            elif self.descent == 'newton':
                beta = _newton(self.distr, self.descent, 
                               self.lambd, self.alpha, self.lrn_rate, 
                               x, y, beta)
  
            # Accumulate loss depending on task
            if TASK[self.task] == 0:
                loss = _cross_entropy_loss(self.distr, beta, x, y, 
                                              self.lambd, self.alpha)
                L.append(loss)
                  
            elif TASK[self.task] == 1:
                loss = _l2_loss(self.distr, beta, x, y, 
                                   self.lambd, self.alpha)
                L.append(loss)
                
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