from tkinter import *
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from bandwidth_calculator import BandwidthCalculator
from estimation import NormalCore, EDF
from histogram import Histogram
from random_number_generator import SimpleRandomNumberGenerator
from random_variable import NormalRandomVariable, UniformRandomVariable, ExponentialRandomVariable, \
    LaplaceRandomVariable, CauchyRandomVariable, RandomVariable
from smoothed_random_variable import SmoothedRandomVariable


class Application:
    WIDTH: int = 790
    HEIGHT: int = 700

    def __init__(self, master):
        self.canvas_graphics_2 = None
        self.canvas_graphics_1 = None

        # График функции распределения
        self.subplot_distribution_function = None

        # График плотности и гистограммы
        self.subplot_density_histogram = None
        self.run_button = None
        self.m_input = None
        self.bandwidth_input = None
        self.n_input = None
        self.clear_button = None
        self.scale_input = None
        self.location_input = None
        self.distribution_changing = None
        self.master = master
        self.master.resizable(width=False, height=False)
        self.create_widgets()

    def create_widgets(self):

        canvas = Canvas(self.master, width=Application.WIDTH, height=Application.HEIGHT, bg='white')
        canvas.pack()

        shape_input = Frame(self.master, bg='white')
        shape_input.place(x=790, y=300, width=250, relheight=0.9, anchor='e')

        distribution = ('Нормальное', 'Равномерное', 'Экспоненциальное', 'Лапласа', 'Коши')
        distribution_label = Label(shape_input, bg='white', text='Вид распределения:', )
        distribution_label.place(relx=0.25, rely=0.18, anchor='n')
        self.distribution_changing = ttk.Combobox(shape_input, values=distribution, state='readonly')
        self.distribution_changing.current(0)
        self.distribution_changing.place(relx=0.7, rely=0.16, relwidth=0.4, relheight=0.08, anchor='n')

        # Это раскоментируем в том случае если будем менять ядра
        # core = ('Нормальное', 'Епанечникова', 'Коши', 'Треугольное', 'Квадратичное')
        # core_label = Label(shape_input, bg='white', text='Вид ядра:',)
        # core_label.place(relx=0.35, rely=0.165, anchor='n')
        # self.core_changing = ttk.Combobox(shape_input, values=core, state='readonly')
        # self.core_changing.current(0)
        # self.core_changing.place(relx=0.7, rely=0.15, relwidth=0.4, relheight=0.08, anchor='n')

        location_label = Label(shape_input, bg='white', text='location:')
        location_label.place(relx=0.355, rely=0.275, anchor='n')
        self.location_input = Entry(shape_input)
        self.location_input.place(relx=0.7, rely=0.26, relwidth=0.4, relheight=0.08, anchor='n')
        self.location_input.insert(0, 0)

        scale_label = Label(shape_input, bg='white', text='scale:', )
        scale_label.place(relx=0.378, rely=0.37, anchor='n')
        self.scale_input = Entry(shape_input)
        self.scale_input.place(relx=0.7, rely=0.36, relwidth=0.4, relheight=0.08, anchor='n')
        self.scale_input.insert(0, 1)

        n_label = Label(shape_input, bg='white', text='N:')
        n_label.place(relx=0.41, rely=0.49, anchor='n')
        self.n_input = Entry(shape_input)
        self.n_input.place(relx=0.7, rely=0.47, relwidth=0.4, relheight=0.08, anchor='n')
        self.n_input.insert(0, 100)

        bandwidth_label = Label(shape_input, bg='white', text='bandwidth:')
        bandwidth_label.place(relx=0.32, rely=0.6, anchor='n')
        self.bandwidth_input = Entry(shape_input)
        self.bandwidth_input.place(relx=0.7, rely=0.58, relwidth=0.4, relheight=0.08, anchor='n')
        self.bandwidth_input.configure(state='disabled')

        m_label = Label(shape_input, bg='white', text='m:')
        m_label.place(relx=0.41, rely=0.7, anchor='n')
        self.m_input = Entry(shape_input)
        self.m_input.place(relx=0.7, rely=0.69, relwidth=0.4, relheight=0.08, anchor='n')
        self.m_input.insert(0, 20)

        self.run_button = Button(shape_input, text='Запустить', command=self.begin)
        self.run_button.place(relx=0.3, rely=0.85, relwidth=0.3, relheight=0.1, anchor='n')

        self.clear_button = Button(shape_input, text='Очистить', command=self.reset)
        self.clear_button.place(relx=0.7, rely=0.85, relwidth=0.3, relheight=0.1, anchor='n')

        image_graphics_1 = Frame(self.master, bg='white')
        image_graphics_1.place(x=500, y=150, width=480, relheight=0.45, anchor='e')
        image_graphics_2 = Frame(self.master, bg='white')
        image_graphics_2.place(x=500, y=480, width=480, relheight=0.45, anchor='e')

        f_1 = Figure()
        self.subplot_distribution_function = f_1.add_subplot(111)
        self.subplot_distribution_function.set_title('График функции распределения')
        self.subplot_distribution_function.grid(True)
        self.canvas_graphics_1 = FigureCanvasTkAgg(f_1, image_graphics_1)
        self.canvas_graphics_1.get_tk_widget().place(relheight=1, relwidth=1)
        self.canvas_graphics_1.draw()

        f_2 = Figure()
        self.subplot_density_histogram = f_2.add_subplot(111)
        self.subplot_density_histogram.set_title('График плотности и гистограммы')
        self.subplot_density_histogram.grid(True)
        self.canvas_graphics_2 = FigureCanvasTkAgg(f_2, image_graphics_2)
        self.canvas_graphics_2.get_tk_widget().place(relheight=1, relwidth=1)
        self.canvas_graphics_2.draw()

    def change_random_variable(self, location, scale) -> RandomVariable:
        if self.distribution_changing.get() == 'Нормальное':
            return NormalRandomVariable(location, scale)
        elif self.distribution_changing.get() == 'Равномерное':
            return UniformRandomVariable(location, scale)
        elif self.distribution_changing.get() == 'Экспоненциальное':
            return ExponentialRandomVariable(scale)
        elif self.distribution_changing.get() == 'Лапласа':
            return LaplaceRandomVariable(location, scale)
        elif self.distribution_changing.get() == 'Коши':
            return CauchyRandomVariable(location, scale)

    def plot_1(self, xs, ys, colors):
        for x, y, c in zip(xs, ys, colors):
            self.subplot_distribution_function.plot(x, y, c)
        self.canvas_graphics_1.draw()

    def plot_2(self, xs, ys, colors):
        for x, y, c in zip(xs, ys, colors):
            self.subplot_density_histogram.plot(x, y, c)
        self.canvas_graphics_2.draw()

    # def change_core(self):
    #     kernel = None
    #     if self.core_changing.get() == 'Нормальное':
    #         kernel = NormalCore()
    #     elif self.core_changing.get() == 'Епанечникова':
    #         kernel = EpanechnikovaCore()
    #     elif self.core_changing.get() == 'Коши':
    #         kernel = CauchyCore()
    #     elif self.core_changing.get() == 'Треугольное':
    #         kernel = TriangularCore()
    #     elif self.core_changing.get() == 'Квадратичное':
    #         kernel = QuadraticCore()
    #     return kernel

    def begin(self):
        self.reset()

        location = int(self.location_input.get())
        scale = int(self.scale_input.get())
        n = int(self.n_input.get())
        points = 100
        m = int(self.m_input.get())

        rv = self.change_random_variable(location, scale)
        generator = SimpleRandomNumberGenerator(rv)

        sample = generator.get(n)

        #  linspace создает массив чисел, равномерно распределенных в интервале между начальным и конечным значениями
        #  3-ий аргумент кол-во точек
        x = np.linspace(np.min(sample), np.max(sample), points)

        # массив вероятностей в точках из массива x.
        y_truth = np.vectorize(rv.cdf)(x)

        edf = EDF(sample)
        # содержит оценки функции распределения, рассчитанные с использованием эмпирической функции распределения, для каждого элемента массива x.
        y_edf = np.vectorize(edf.value)(x)

        core = NormalCore()  # self.change_core()
        bandwidth_calculator = BandwidthCalculator(sample, core)
        bandwidth = bandwidth_calculator.calculate_bandwidth()

        self.bandwidth_input.configure(state='normal')
        self.bandwidth_input.delete(0, END)
        self.bandwidth_input.insert(0, bandwidth)
        self.bandwidth_input.configure(state='disabled')

        srv = SmoothedRandomVariable(sample, core, bandwidth)
        y_kernel = np.vectorize(srv.cdf)(x)

        p_1 = np.vectorize(rv.pdf)(x)

        hist = Histogram(sample, m)
        p_2 = np.vectorize(hist.value)(x)

        p_3 = np.vectorize(srv.pdf)(x)

        self.plot_1([x] * 3, [y_truth, y_edf, y_kernel], ['red', 'blue', 'green'])
        self.plot_2([x] * 3, [p_1, p_2, p_3], ['red', 'blue', 'g'])

    def reset(self):
        self.subplot_distribution_function.clear()
        self.subplot_distribution_function.set_title('График функции распределения')
        self.subplot_distribution_function.grid(True)
        self.canvas_graphics_1.draw()

        self.subplot_density_histogram.clear()
        self.subplot_density_histogram.set_title('График плотности и гистограммы')
        self.subplot_density_histogram.grid(True)
        self.canvas_graphics_2.draw()
