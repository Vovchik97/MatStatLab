from abc import ABC, abstractmethod

import numpy as np
from random_variable import RandomVariable


class RandomNumberGenerator(ABC):
    def __init__(self, random_variable: RandomVariable):
        self.random_variable = random_variable

    @abstractmethod
    def get(self, N):
        pass


class SimpleRandomNumberGenerator(RandomNumberGenerator):
    def __init__(self, random_variable: RandomVariable):
        super().__init__(random_variable)

    def get(self, N: int) -> np.ndarray:
        us = np.random.uniform(0, 1, N)
        return np.vectorize(self.random_variable.quantile)(us)
