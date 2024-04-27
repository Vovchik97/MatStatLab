import numpy as np

from estimation import Estimation
from random_variable import RandomVariable


class SmoothedRandomVariable(RandomVariable, Estimation):
    def __init__(self, sample, core, bandwidth):
        super().__init__(sample)
        self.core = core
        self.h = bandwidth

    def pdf(self, x):
        return np.mean([self.core._k((x - y) / self.h) for y in self.sample]) / self.h

    def cdf(self, x):
        return np.mean([self.core._K((x - y) / self.h) for y in self.sample])

    def quantile(self, alpha):
        raise NotImplementedError