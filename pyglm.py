# A python implementation of GLMs #

import numpy as np

def _mu(dist, X, beta):
    '''Computes the response function for different distributions

    Parameters
    ----------
    dist : str
        Name of the distribution. Choices:
        gaussian | poisson | inv_gaussian | exponential | multinomial
        
    beta : array-like, shape = [n features]
                Vector of weights
    X : array-like, shape = [n_samples, n_features]
            Vector of training data
    
    Returns
    -------
    mu : float
        Expectation value of the distribution function for a single sample
        '''
    if dist == 'gaussian':
        mu = np.dot(X, beta)
        return mu

    elif dist == 'poisson':
        nu = np.dot(X, beta)
        mu = np.exp(nu)
        return mu

    elif dist == 'inv_gaussian':
        nu = np.dot(X,beta)
        mu = np.sqrt(nu)
        return mu

    elif dist == 'exponential':
        nu = np.dot(X, beta)
        mu = 1 / nu
        return mu

    elif dist == 'multinomial':
        nu = np.dot(X, beta)
        mu = 1 / (1 + np.exp(-nu))
        return mu

def _inverse_hessian(X, y, beta, n_samples, n_features):
    '''Computes the Inverse of the Hessian of the log-likelihood function

    X : array-like, shape = [n_samples, n_features]
        Vector of training data

    y : array-like, shape = [n_samples]
        Vector of response data

    beta : array-like, shape = [n features]
        Vector of weights

    n_samples : int
        Number of training examples

    n_features : int
        Number of features (including intercept)

    Returns
    -------
    inv_hessian : array-like, shape = [n_features, n_features]
       Inverse of the hessian for the loss function
    '''
    hessian = np.dot(np.transpose(X), X)    
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


def _loss(dist, descent, lambd, alpha, X, y, beta, n_samples, n_features):
    '''Computes the l2 regularized loss function

    Parameters
    ----------
    dist : str
        Name of the distribution. Choices:
        gaussian | poisson | inv_gaussian | exponential | multinomial
    
    descent : str
        Fitting method used:
        b-gradient | s-gradient | newton
        
    lambd : float
        General regularization parameter
            
    alpha : float
        Elastic net regularization parameter
        alpha = 0   
        
    X : array-like, shape = [n_samples, n_features]
        Vector of training data

    y : array-like, shape = [n_samples]
        Vector of response data

    beta : array-like, shape = [n features]
        Vector of weights

    n_samples : int
        Number of training examples

    n_features : int
        Number of features (including intercept)

    Returns
    -------
    loss : int
        Computed loss 
    '''
    loss = np.zeros(n_features)
    gradient = np.zeros(n_features)

    if descent == 'b-gradient':
        for i in range(0, n_features):
            if i == 0:
                loss[i] = (1/n_samples) * sum((_mu(dist, X, beta) - y) * X[:,i])
            elif i != 0:
                loss[i] = (1/(2 * n_samples)) * (sum((_mu(dist, X, beta) - y) * X[:,i]) + _reg(lambd, alpha, beta[i]))

    elif descent == 's-gradient':
        if n_samples == 0:
            loss = (_mu(dist, X, beta) - y) * X[n_features]
        elif n_samples != 0:
            loss = (_mu(dist, X, beta) - y) * X[n_features] + _reg(lambd, alpha, beta[n_features])

    elif descent == 'newton':      
        for i in range(0, n_features):
            gradient[i] = (1/n_samples) * sum((_mu(dist, X, beta) - y) * X[:,i])
        inv_hessian = _inverse_hessian(X, y, beta, n_samples, n_features)
        loss = np.dot(inv_hessian, gradient)

    return loss


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

    def __init__(self, dist='gaussian', n_iter=100, descent='b-gradient', learning_rate=1,
                    lambd=1, alpha=0.5):
        '''

        Attributes
        ----------
        dist : str
            Name of the distribution. Choices:
            gaussian | poisson | inv_gaussian | exponential | multinomial

        n_iter : integer
            Number of iterations of to compute cost function

        descent : str
            Fitting method used:
            b-gradient | s-gradient | newton

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

        self.dist = dist
        self.n_iter = n_iter
        self.descent = descent
        self.learning_rate = learning_rate
        self.lambd = lambd
        self.alpha = alpha

    
    
    def fit(self, X, y):
        '''Computes the optimal weights that minimize the cost function

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Vector of training data
        y : array-like, shape = [n_samples]
            Vector of response data

        Returns
        -------
        beta : array-like, shape = [n features]
            Vector of optimized weights   
        '''
        
        n_samples = np.shape(X)[0]
        intercept = np.ones(n_samples)
        X = np.vstack((np.transpose(intercept), X))
        X = np.transpose(X)
        
        n_features = np.shape(X)[1]
        beta = np.zeros(n_features)

        if self.descent == 'b-gradient':
            for i in range(0, self.n_iter):
                loss = _loss(self.dist, self.descent, self.lambd, self.alpha, X, y, beta, n_samples, n_features)
                beta = beta - self.learning_rate * loss

        elif self.descent == 's-gradient':
            for i in range(0, n_samples):
                for j in range(0, n_features):
                    loss = _loss(self.dist, self.descent, self.lambd, self.alpha, X[i,:], y[i], beta, i, j)
                    beta = beta - self.learning_rate * loss
        
        elif self.descent == 'newton':
            for i in range(0, self.n_iter):
                loss = _loss(self.dist, self.descent, self.lambd, self.alpha, X, y, beta, n_samples, n_features)
                beta = beta - self.learning_rate * loss
                
        return beta

    
    def predict(self, X, beta):
        '''Predicts response variables using the optimal weights

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Vector of training data
            
        beta : array-like, shape = [n features]
            Vector of optimized weights  

        Returns
        -------
        prediction : array-like, shape = [n_samples]
            Vector of predicted response
        '''
        prediction = np.dot(X, beta)
        
        return prediction
    
    


        

        
            
        

            
        
        
        

    
