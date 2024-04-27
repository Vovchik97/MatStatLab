import numpy as np

from estimation import Estimation


class Histogram(Estimation):
    def __init__(self, sample, m):
        super().__init__(sample)
        self.sub_interval_width = None
        self.intervals = None
        self.m = m
        self.init_intervals(sample)  # Передаем sample в функцию init_intervals

    class Interval:
        def __init__(self, a: float, b: float):
            self.a = a  # Левая граница интервала.
            self.b = b  # Правая граница интервала.

        def is_in(self, x):
            """
            Проверяет, находится ли значение x внутри данного интервала.
            """
            return self.a <= x <= self.b

        def __repr__(self):
            """
            Возвращает строковое представление интервала.
            """
            return f'({self.a}, {self.b})'

    def init_intervals(self, sample):
        """
        Инициализирует интервалы гистограммы на основе выборки данных sample.
        """

        # Генерируем левые границы интервалов гистограммы с использованием линейного разбиения от минимального до максимального значения выборки.
        # Для этого используем функцию linspace из библиотеки numpy, которая разбивает заданный интервал на равные части.
        # Количество частей определяется параметром m, который указывает на количество интервалов.
        left_boundary_of_intervals = np.linspace(np.min(sample), np.max(sample), self.m + 1)[:-1]

        # Вычисляем правые границы интервалов гистограммы.
        # Для этого конкатенируем (соединяем) левые границы интервалов, начиная со второго интервала, и добавляем максимальное значение выборки.
        # Это позволяет получить правые границы для каждого интервала.
        right_boundary_of_intervals = np.concatenate((left_boundary_of_intervals[1:], [np.max(sample)]))

        # Создаем список объектов класса Interval, представляющих собой интервалы гистограммы.
        # Каждый интервал определяется левой и правой границами, которые берутся из соответствующих списков левых и правых границ интервалов.
        self.intervals = [Histogram.Interval(a, b) for a, b in
                          zip(left_boundary_of_intervals, right_boundary_of_intervals)]

        # Вычисляем ширину подинтервала гистограммы, которая равна разности первой правой и первой левой границ интервалов.
        # Это значение используется для нормировки гистограммы при вычислении ее значений в точках.
        self.sub_interval_width = right_boundary_of_intervals[0] - left_boundary_of_intervals[0]

    def get_interval(self, x):
        """
        Возвращает интервал гистограммы, в который попадает значение x.
        """
        for i in self.intervals:
            if i.is_in(x):
                return i
        return None

    def get_sample_by_interval(self, interval):
        return np.array(list(filter(lambda x: interval.is_in(x), self.sample)))

    def value(self, x):
        return len(self.get_sample_by_interval(self.get_interval(x))) / (self.sub_interval_width * len(self.sample))
