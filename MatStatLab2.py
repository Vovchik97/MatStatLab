import math
import numpy as np
from abc import ABC, abstractmethod
from tkinter import *
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# RandomVariable
class RandomVariable(ABC):
    @abstractmethod
    def pdf(self, x):
        pass

    @abstractmethod
    def cdf(self, x):
        pass

    @abstractmethod
    def quantile(self, alpha):
        pass

# Нормальное распределение
class NormalRandomVariable(RandomVariable):
    def __init__(self, location = 0, scale = 1) -> None:
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

'''Эти методы определяют поведение различных типов случайных величин, таких как нормальное распределение (NormalRandomVariable), 
равномерное распределение (UniformRandomVariable), 
экспоненциальное распределение (ExponentialRandomVariable) 
и распределение Лапласа (LaplaceRandomVariable). 
Каждый тип случайной величины имеет методы для вычисления функции плотности вероятности (pdf), функции распределения (cdf) и квантили (quantile).

Функция плотности вероятности (pdf) используется для описания вероятностного распределения значений случайной величины. 
Она определяет вероятность того, что случайная величина примет значение x.

Функция распределения (cdf) определяет вероятность того, 
что случайная величина будет меньше или равна определенному значению x.

Квантиль (quantile) является обратной функцией распределения и используется для нахождения значения x, 
при котором случайная величина примет определенную вероятность alpha.

Эти методы важны для проведения статистического анализа данных и построения соответствующих графиков распределений.'''

def Change_RandomVariable(location, scale):
    if distribution_changing.get() == 'Нормальное':
        return NormalRandomVariable(location, scale)
    elif distribution_changing.get() == 'Равномерное':
        return UniformRandomVariable(location, scale)
    elif distribution_changing.get() == 'Экспоненциальное':
        return ExponentialRandomVariable(scale)
    elif distribution_changing.get() == 'Лапласса':
        return LaplaceRandomVariable(location, scale)
    elif distribution_changing.get() == 'Коши':
        return CauchyRandomVariable(location, scale)

# RandomNumberGenerator
class RandomNumberGenerator(ABC):
    def __init__(self, random_variable: RandomVariable):
        self.random_variable = random_variable

    @abstractmethod
    def get(self, N):
        pass

class SimpleRandomNumberGenerator(RandomNumberGenerator):
    def __init__(self, random_variable: RandomVariable):
        super().__init__(random_variable)

    def get(self, N):
        us = np.random.uniform(0, 1, N)
        return np.vectorize(self.random_variable.quantile)(us)

def plot_1(xs, ys, colors):
    for x, y, c in zip(xs, ys, colors):
        graphics_1.plot(x, y, c)
    canvas_graphics_1.draw()

def plot_2(xs, ys, colors):
    for x, y, c in zip(xs, ys, colors):
        graphics_2.plot(x, y, c)
    canvas_graphics_2.draw()

# Estimation
class Estimation(ABC):
    def __init__(self, sample):
        self.sample = sample

''' Класс EDF представляет оценку функции распределения на основе эмпирической функции распределения 
(empirical distribution function), 
которая вычисляет долю значений в выборке, меньших или равных данному значению x.'''
class EDF(Estimation):
    def heaviside_function(x):
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

'''Классы NormalCore, EpanechnikovaCore, CauchyCore, TriangularCore и QuadraticCore представляют различные ядра, 
которые могут использоваться при оценке плотности вероятности на основе KDE.
Ядра (или ядерные функции) используются в методе kernel density estimation (KDE), 
который является одним из методов оценки плотности распределения случайных величин. 
Ядро представляет собой функцию, которая определяет вес, 
который будет присвоен каждому элементу выборки при расчете оценки плотности вероятности на основе метода KDE.'''
class NormalCore(Function):
    def _k(self, x):
        return math.exp(-0.5 * x ** 2) / (math.sqrt(2 * math.pi))

    def _K(self, x):
        if x <= 0:
            return 0.852 * math.exp(-math.pow((-x + 1.5774) / 2.0637, 2.34))
        return 1 - 0.852 * math.exp(-math.pow((x + 1.5774) / 2.0637, 2.34))

    def h(self, x):
        return -x * self._k(x)


class EpanechnikovaCore(Function):
    def _k(self, x):
        if abs(x) <= 1:
            return 0.75 * (1 - x ** 2)
        else:
            return 0

    def _K(self, x):
        if x < -1:
            return 0
        elif x <= 1:
            return (0.75 * x) + (0.5 * x ** 3) + (1 / 3)
        else:
            return 1

    def h(self, x):
        if abs(x) <= 1:
            return -1.5 * x
        else:
            return 0


class CauchyCore(Function):
    def _k(self, x):
        return 1 / (math.pi * (1 + x ** 2))

    def _K(self, x):
        return math.atan(x) / math.pi + 0.5

    def h(self, x):
        return -2 * x / (math.pi * (1 + x ** 2))


class TriangularCore(Function):
    def _k(self, x):
        if abs(x) <= 1:
            return 1 - abs(x)
        else:
            return 0

    def _K(self, x):
        if x < -1:
            return 0
        elif x <= 1:
            return 0.5 * (1 - abs(x)) ** 2
        else:
            return 1

    def h(self, x):
        if x < -1:
            return 0
        elif -1 <= x < 0:
            return 1
        elif 0 <= x <= 1:
            return -1
        else:
            return 0


class QuadraticCore(Function):
    def _k(self, x):
        if abs(x) <= 1:
            return 0.75 * (1 - x ** 2)
        else:
            return 0

    def _K(self, x):
        if x < -1:
            return 0
        elif x <= 1:
            return 0.5 * (x + 1)
        else:
            return 1

    def h(self, x):
        if abs(x) <= 1:
            return -2 * x
        else:
            return 0


def Change_Core():
    if core_changing.get() == 'Нормальное':
        kernel = NormalCore()
    elif core_changing.get() == 'Епанечникова':
        kernel = EpanechnikovaCore()
    elif core_changing.get() == 'Коши':
        kernel = CauchyCore()
    elif core_changing.get() == 'Треугольное':
        kernel = TriangularCore()
    elif core_changing.get() == 'Квадратичное':
        kernel = QuadraticCore()
    return kernel

'''Класс SmoothedRandomVariable принимает выборку значений случайной величины и использует метод ядерной оценки для оценки плотности вероятности. 
Класс определяет параметр h, который является шириной окна, используемого при оценке плотности вероятности. 
Значение ширины окна вычисляется автоматически на основе выборки в методе get_h(). 
В этом методе используется алгоритм скользящего бутстрэпа (bootstrap) для выбора оптимального значения ширины окна.'''
class SmoothedRandomVariable(RandomVariable, Estimation):
    def __init__(self, sample):
        super().__init__(sample)
        self.core = Change_Core()
        self.h = self.get_h()

    def get_h(self):
        n = len(self.sample)
        temp = 0.0
        for i in range(n):
            temp += self.sample[i]
        mean = temp / n
        temp2 = 0.0
        for i in range(n):
            temp2 += (self.sample[i] - mean) ** 2
        hn = math.sqrt(temp2) / (n - 1)
        delta = 1.0
        while (delta >= 0.001):
            s = 0.0
            for i in range(n):
                num = 0.0
                div = 0.0
                for j in range(n):
                    if i != j:
                        diff = (self.sample[j] - self.sample[i]) / hn
                        num += self.core.h(diff) * (self.sample[j] - self.sample[i])
                        div += self.core._k(diff)
                if div == 0.0:
                    continue
                s += num / div
            new_hn = - (1 / n) * s
            delta = abs(new_hn - hn)
            hn = new_hn
        bandwidth_input.configure(state='normal')
        bandwidth_input.delete(0, END)
        bandwidth_input.insert(0, hn)
        bandwidth_input.configure(state='disabled')
        return hn

    def pdf(self, x):
        return np.mean([self.core._k((x - y) / self.h) for y in self.sample]) / self.h

    def cdf(self, x):
        return np.mean([self.core._K((x - y) / self.h) for y in self.sample])

    def quantile(self, alpha):
        raise NotImplementedError


'''Класс Histogram также принимает выборку значений случайной величины и использует метод гистограммы для оценки плотности вероятности. 
Класс определяет параметр m, который является количеством интервалов гистограммы. 
Значения границ интервалов вычисляются автоматически на основе выборки в методе init_intervals(). 
В этом методе выборка разбивается на m равных интервалов, и каждый интервал соответствует некоторому значению плотности вероятности.'''
class Histogram(Estimation):
    class Interval:
        def __init__(self, a, b):
            self.a = a
            self.b = b

        def is_in(self, x):
            return x >= self.a and x <= self.b

        def __repr__(self):
            return f'({self.a}, {self.b})'

    def __init__(self, sample, m):
        super().__init__(sample)
        self.m = m

        self.init_intervals()

    def init_intervals(self):
        left_boundary_of_intervals = np.linspace(np.min(sample), np.max(sample), self.m + 1)[:-1]
        right_boundary_of_intervals = np.concatenate((left_boundary_of_intervals[1:], [np.max(sample)]))

        self.intervals = [Histogram.Interval(a, b) for a, b in
                          zip(left_boundary_of_intervals, right_boundary_of_intervals)]

        self.sub_interval_width = right_boundary_of_intervals[0] - left_boundary_of_intervals[0]

    def get_interval(self, x):
        for i in self.intervals:
            if i.is_in(x):
                return i
        return None

    def get_sample_by_interval(self, interval):
        return np.array(list(filter(lambda x: interval.is_in(x), self.sample)))

    def value(self, x):
        return len(self.get_sample_by_interval(self.get_interval(x))) / (self.sub_interval_width * len(self.sample))

def begin():
    reset()

    location = int(location_input.get())
    scale = int(scale_input.get())
    N = int(N_input.get())
    M = 100
    m = int(m_input.get())

    rv = Change_RandomVariable(location, scale)
    generator = SimpleRandomNumberGenerator(rv)

    global sample
    sample = generator.get(N)

    X = np.linspace(np.min(sample), np.max(sample), M)

    Y_truth = np.vectorize(rv.cdf)(X)

    edf = EDF(sample)
    Y_edf = np.vectorize(edf.value)(X)

    srv = SmoothedRandomVariable(sample)
    Y_kernel = np.vectorize(srv.cdf)(X)

    P_1 = np.vectorize(rv.pdf)(X)

    hist = Histogram(sample, m)
    P_2 = np.vectorize(hist.value)(X)

    P_3 = np.vectorize(srv.pdf)(X)
    '''На первом графике используются цвета red для истинной функции распределения, blue для EDF и green для KDE. 
       На втором графике используется цвет red для PDF, blue для гистограммы и зеленый для сглаженной случайной величины.'''
    plot_1([X] * 3, [Y_truth, Y_edf, Y_kernel], ['red', 'blue', 'green'])
    plot_2([X] * 3, [P_1, P_2, P_3], ['red', 'blue', 'g'])

'''Первый график отображает функцию распределения, которая показывает вероятность того, 
что случайная величина примет значение, меньшее или равное определенному значению. 
На этом графике можно увидеть, как истинная функция распределения (красная линия) сравнивается с эмпирической функцией распределения (синяя линия), полученной из выборки. 
Также на графике представляется ядерная оценка плотности (зеленая линия), которая используется для аппроксимации функции распределения.

Второй график отображает плотность вероятности распределения, которая показывает вероятность того, 
что случайная величина примет значение в определенном диапазоне. 
Также на графике представлена гистограмма (синие столбцы), которая показывает частоту, 
с которой значения случайной величины выпадали в различных диапазонах. 
Красная линия на графике представляет собой плотность вероятности, соответствующую использованной модели распределения. 
Зеленая линия - это сглаженная случайная величина, полученная из выборки. 
Этот график также используется для сравнения модели распределения и данных из выборки, что помогает определить, насколько хорошо модель подходит к данным.'''

def reset():
    global graphics_1
    global graphics_2

    graphics_1.clear()
    graphics_1.set_title('График функции распределения')
    graphics_1.grid(True)
    canvas_graphics_1.draw()

    graphics_2.clear()
    graphics_2.set_title('График плотности и гистограммы')
    graphics_2.grid(True)
    canvas_graphics_2.draw()

root = Tk()
root.title("Лабораторная работа №2")
root.resizable(width=False, height=False)
WIDTH = 790
HEIGHT = 700
canvas = Canvas(root, width=WIDTH, height=HEIGHT, bg='lightblue')
canvas.pack()

shape_input = Frame(root, bg='#ffa70f')
shape_input.place(x=790, y=300, width=250, relheight=0.9, anchor='e')

distribution = ('Нормальное', 'Равномерное', 'Экспоненциальное', 'Лапласса', 'Коши')
distribution_label = Label(shape_input, bg='#ffa70f', text='Вид распределения:', fg='white')
distribution_label.place(relx=0.25, rely=0.05, anchor='n')
distribution_changing = ttk.Combobox(shape_input, values=distribution, state='readonly')
distribution_changing.current(0)
distribution_changing.place(relx=0.7, rely=0.04, relwidth=0.4, relheight=0.08, anchor='n')

core = ('Нормальное', 'Епанечникова', 'Коши', 'Треугольное', 'Квадратичное')
core_label = Label(shape_input, bg='#ffa70f', text='Вид ядра:', fg='white')
core_label.place(relx=0.35, rely=0.165, anchor='n')
core_changing = ttk.Combobox(shape_input, values=core, state='readonly')
core_changing.current(0)
core_changing.place(relx=0.7, rely=0.15, relwidth=0.4, relheight=0.08, anchor='n')

location_label = Label(shape_input, bg='#ffa70f', text='location:', fg='white')
location_label.place(relx=0.355, rely=0.275, anchor='n')
location_input = Entry(shape_input)
location_input.place(relx=0.7, rely=0.26, relwidth=0.4, relheight=0.08, anchor='n')
location_input.insert(0, 0)

scale_label = Label(shape_input, bg='#ffa70f', text='scale:', fg='white')
scale_label.place(relx=0.378, rely=0.37, anchor='n')
scale_input = Entry(shape_input)
scale_input.place(relx=0.7, rely=0.36, relwidth=0.4, relheight=0.08, anchor='n')
scale_input.insert(0, 1)

N_label = Label(shape_input, bg='#ffa70f', text='N:', fg='white')
N_label.place(relx=0.41, rely=0.49, anchor='n')
N_input = Entry(shape_input)
N_input.place(relx=0.7, rely=0.47, relwidth=0.4, relheight=0.08, anchor='n')
N_input.insert(0, 100)

bandwidth_label = Label(shape_input, bg='#ffa70f', text='bandwidth:', fg='white')
bandwidth_label.place(relx=0.32, rely=0.6, anchor='n')
bandwidth_input = Entry(shape_input)
bandwidth_input.place(relx=0.7, rely=0.58, relwidth=0.4, relheight=0.08, anchor='n')
bandwidth_input.configure(state='disabled')

m_label = Label(shape_input, bg='#ffa70f', text='m:', fg='white')
m_label.place(relx=0.41, rely=0.7, anchor='n')
m_input = Entry(shape_input)
m_input.place(relx=0.7, rely=0.69, relwidth=0.4, relheight=0.08, anchor='n')
m_input.insert(0, 20)

run_button = Button(shape_input, text='Запустить', command=begin)
run_button.place(relx=0.3, rely=0.85, relwidth=0.3, relheight=0.1, anchor='n')

clear_button = Button(shape_input, text='Очистить', command=reset)
clear_button.place(relx=0.7, rely=0.85, relwidth=0.3, relheight=0.1, anchor='n')

image_graphics_1 = Frame(root, bg='white')
image_graphics_1.place(x=500, y=150, width=480, relheight=0.45, anchor='e')
image_graphics_2 = Frame(root, bg='white')
image_graphics_2.place(x=500, y=480, width=480, relheight=0.45, anchor='e')

f_1 = Figure()
graphics_1 = f_1.add_subplot(111)
graphics_1.set_title('График функции распределения')
graphics_1.grid(True)
canvas_graphics_1 = FigureCanvasTkAgg(f_1, image_graphics_1)
canvas_graphics_1.get_tk_widget().place(relheight=1, relwidth=1)
canvas_graphics_1.draw()

f_2 = Figure()
graphics_2 = f_2.add_subplot(111)
graphics_2.set_title('График плотности и гистограммы')
graphics_2.grid(True)
canvas_graphics_2 = FigureCanvasTkAgg(f_2, image_graphics_2)
canvas_graphics_2.get_tk_widget().place(relheight=1, relwidth=1)
canvas_graphics_2.draw()

root.mainloop()