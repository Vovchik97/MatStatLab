import math
from abc import ABC, abstractmethod

import numpy as np


class Estimation(ABC):
    def __init__(self, sample):
        self.sample = sample


class EDF(Estimation):
    def heaviside_function(self, x):
        if x > 0:
            return 1
        else:
            return 0

    def value(self, x):
        return np.mean(np.vectorize(EDF.heaviside_function)(x - self.sample))


class Function(ABC):
    @abstractmethod
    def _k(self, x):
        pass

    @abstractmethod
    def _K(self, x):
        pass

    @abstractmethod
    def h(self, x):
        pass


class NormalCore(Function):
    def _k(self, x):
        return math.exp(-0.5 * x ** 2) / (math.sqrt(2 * math.pi))

    def _K(self, x):
        if x <= 0:
            return 0.852 * math.exp(-math.pow((-x + 1.5774) / 2.0637, 2.34))
        return 1 - 0.852 * math.exp(-math.pow((x + 1.5774) / 2.0637, 2.34))

    def h(self, x):
        return -x * self._k(x)


# class EpanechnikovaCore(Function):
#     def _k(self, x):
#         if abs(x) <= 1:
#             return 0.75 * (1 - x ** 2)
#         else:
#             return 0
#
#     def _K(self, x):
#         if x < -1:
#             return 0
#         elif x <= 1:
#             return (0.75 * x) + (0.5 * x ** 3) + (1 / 3)
#         else:
#             return 1
#
#     def h(self, x):
#         if abs(x) <= 1:
#             return -1.5 * x
#         else:
#             return 0
#
#
# class CauchyCore(Function):
#     def _k(self, x):
#         return 1 / (math.pi * (1 + x ** 2))
#
#     def _K(self, x):
#         return math.atan(x) / math.pi + 0.5
#
#     def h(self, x):
#         return -2 * x / (math.pi * (1 + x ** 2))
#
#
# class TriangularCore(Function):
#     def _k(self, x):
#         if abs(x) <= 1:
#             return 1 - abs(x)
#         else:
#             return 0
#
#     def _K(self, x):
#         if x < -1:
#             return 0
#         elif x <= 1:
#             return 0.5 * (1 - abs(x)) ** 2
#         else:
#             return 1
#
#     def h(self, x):
#         if x < -1:
#             return 0
#         elif -1 <= x < 0:
#             return 1
#         elif 0 <= x <= 1:
#             return -1
#         else:
#             return 0
#
#
# class QuadraticCore(Function):
#     def _k(self, x):
#         if abs(x) <= 1:
#             return 0.75 * (1 - x ** 2)
#         else:
#             return 0
#
#     def _K(self, x):
#         if x < -1:
#             return 0
#         elif x <= 1:
#             return 0.5 * (x + 1)
#         else:
#             return 1
#
#     def h(self, x):
#         if abs(x) <= 1:
#             return -2 * x
#         else:
#             return 0
