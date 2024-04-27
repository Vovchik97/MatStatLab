import math
from abc import ABC, abstractmethod


class RandomVariable(ABC):
    @abstractmethod
    def pdf(self, x):
        """
        Возвращает значение функции плотности вероятности (probability density function) в точке x.
        """
        pass

    @abstractmethod
    def cdf(self, x):
        """
        Возвращает значение функции распределения (cumulative distribution function) в точке x.
        """
        pass

    @abstractmethod
    def quantile(self, alpha):
        """
        Возвращает квантиль уровня alpha, т.е. значение x такое, что P(X <= x) = alpha.
        """
        pass


class NormalRandomVariable(RandomVariable):
    def __init__(self, location=0, scale=1) -> None:
        super().__init__()
        self.location = location
        self.scale = scale

    def pdf(self, x):
        z = (x - self.location) / self.scale
        return math.exp(-0.5 * z * z) / (math.sqrt(2 * math.pi) * self.scale)

    def cdf(self, x):
        z = (x - self.location) / self.scale
        if z <= 0:
            return 0.852 * math.exp(-math.pow((-z + 1.5774) / 2.0637, 2.34))
        return 1 - 0.852 * math.exp(-math.pow((z + 1.5774) / 2.0637, 2.34))

    def quantile(self, alpha):
        return self.location + 4.91 * self.scale * (math.pow(alpha, 0.14) - math.pow(1 - alpha, 0.14))


# Равномерное распределение
class UniformRandomVariable(RandomVariable):
    def __init__(self, left=0, right=1) -> None:
        super().__init__()
        self.left = left
        self.right = right

    def pdf(self, x):
        if x < self.left or x > self.right:
            return 0
        else:
            return 1 / (self.right - self.left)

    def cdf(self, x):
        if x < self.left:
            return 0
        elif x > self.right:
            return 1
        else:
            return (x - self.left) / (self.right - self.left)

    def quantile(self, alpha):
        return self.left + alpha * (self.right - self.left)


# Экспоненциальное распределение
class ExponentialRandomVariable(RandomVariable):
    def __init__(self, scale=1) -> None:
        self.scale = scale

    def pdf(self, x):
        if x < 0:
            return 0
        return self.scale * math.exp(-self.scale * x)

    def cdf(self, x):
        if x < 0:
            return 0
        return 1 - math.exp(-self.scale * x)

    def quantile(self, alpha):
        if alpha < 0 or alpha > 1:
            return None
        return -math.log(1 - alpha) / self.scale


# Распределение Лапласа
class LaplaceRandomVariable(RandomVariable):
    def __init__(self, location=0, scale=1) -> None:
        super().__init__()
        self.location = location
        self.scale = scale

    def pdf(self, x):
        return 0.5 * self.scale * math.exp(-self.scale * abs(x - self.location))

    def cdf(self, x):
        if x < self.location:
            return 0.5 * math.exp((x - self.location) / self.scale)
        else:
            return 1 - 0.5 * math.exp(-(x - self.location) / self.scale)

    def quantile(self, alpha):
        if alpha == 0.5:
            return self.location
        elif alpha < 0.5:
            return self.location - self.scale * math.log(1 - 2 * alpha)
        else:
            return self.location + self.scale * math.log(2 * alpha - 1)


# Распределение Коши
class CauchyRandomVariable(RandomVariable):
    def __init__(self, location=0, scale=1) -> None:
        super().__init__()
        self.location = location
        self.scale = scale

    def pdf(self, x):
        return 1 / (math.pi * self.scale * (1 + ((x - self.location) / self.scale) ** 2))

    def cdf(self, x):
        return 0.5 + (1 / math.pi) * math.atan((x - self.location) / self.scale)

    def quantile(self, alpha):
        return self.location + self.scale * math.tan(math.pi * (alpha - 0.5))
