import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError


class RFF(BaseEstimator):
    """
        Random Fourier Features.
    """

    def __init__(self, gamma=1., n_components=100):
        """
            gamma: Gaussian distribution parameter
            n_components: number of samples drawn from Gaussian distribution
            fitted: boolean parameter to avoid transformation before fitting
        """
        self.gamma = gamma
        self.n_components = n_components
        self.fitted = False


    def fit(self, X, y=None):
        """
            Input: X with shape (num_samples, num_features)
            Output: self, the fitted transformer
        """
        X = np.array(X)
        num_features = X.shape[1]

        self.weight = np.sqrt(2*self.gamma) * np.random.normal(size=(self.n_components, num_features))
        self.bias = np.random.uniform(0, 2 * np.pi, size=self.n_components)
        self.fitted = True
        return self


    def transform(self, X):
        """
            Input: X with shape (num_samples, num_features)
            Output: new X with shape (num_samples, n_components)
        """
        X = np.array(X)

        if not self.fitted:
            raise NotFittedError('Fourier feature should be fitted before transforming')

        new_X = np.sqrt(2/self.n_components) * np.cos((X.dot(self.weight.T) + self.bias[np.newaxis, :]))
        return new_X



class RFF_sincos(RFF):
    """
        Random Fourier Features uses not only cos but also sin for random features.
    """

    def fit(self, X, y=None):
        """
            Input: X with shape (num_samples, num_features)
            Output: self, the fitted transformer
        """
        X = np.array(X)
        num_features = X.shape[1]

        self.weight = np.sqrt(2*self.gamma) * np.random.normal(size=(int(self.n_components/2), num_features))
        self.fitted = True
        return self


    def transform(self, X):
        """
            Input: X with shape (num_samples, num_features)
            Output: new X with shape (num_samples, n_components)
        """
        X = np.array(X)

        if not self.fitted:
            raise NotFittedError('Fourier feature should be fitted before transforming')

        X_w = X.dot(self.weight.T)
        new_X = np.sqrt(2/self.n_components) * np.concatenate([np.cos(X_w), np.sin(X_w)], axis=1)
        return new_X