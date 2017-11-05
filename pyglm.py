# A python implementation of various Generalized Linear Models

import numpy as np

class GLM():
    '''This class defines a generalized linear-model

    Parameters
    ----------

    Attributes
    ----------

    '''

    def __init__(self, distribution):
        '''

        Parameters
        ----------
        distribution : string
            Name of the distribution. Possible choises.
            gaussian | poisson | inv_gaussian | gamma | multinomial

        Returns
        -------


        '''

        self.distribution = distribution


        return

    def fit(self, X, y):
        '''

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Vector of training data
        y : array-like, shape = [n_samples]
            Vector of response data

        Returns
        -------

        '''

        self.X = X
        self.y = y

        return

    def predict(self, X):
        '''

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Vector of training data

        Returns
        -------

        '''

        return