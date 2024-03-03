from random import randint  # Импорт функции randint из модуля random
import numpy as np
import matplotlib.pyplot as plt
import math
import tkinter as tk
from tkinter import ttk  # Импорт класса ttk из модуля tkinter

# Функция для моделирования броска монеты
def flip_coin():
    return randint(0, 1)

# Функция для выполнения одного эксперимента (бросания монеты)
def experiment(N):
    result = np.zeros(N)  # Создание массива нулей размером N
    counter = 0
    for i in range(N):
        fc = flip_coin()  # Выполнение броска монеты
        if fc == 1:
            counter += 1
        result[i] = counter / (i + 1)  # Расчет доли выпавших орлов к общему количеству бросков
    return result

# Функция для выполнения серии экспериментов
def exp_serial(M, N):
    result = np.zeros((M, N))  # Создание двумерного массива нулей размером M x N
    for i in range(M):
        result[i,] = experiment(N)  # Выполнение серии экспериментов
    return result

# Функция для расчета среднего значения массива
def mean(vs):
    return np.mean(vs, axis=0)  # Вычисление среднего значения по столбцам

# Функция для вычисления доверительного интервала
def conf_interval(vs, alpha):
    m = vs.shape[0]  # Количество строк в массиве
    a = (1 - alpha) / 2  # Вероятность альфа
    m_down = int(m * a)  # Нижняя граница доверительного интервала
    m_up = m - m_down - 1  # Верхняя граница доверительного интервала

    sorted_vs = np.sort(vs, axis=0)  # Сортировка значений в каждом столбце

    return np.apply_along_axis(lambda x: np.array([x[m_down], x[m_up]]), 0, sorted_vs)

# Функция для вычисления квантиля нормального распределения
def normal_quantile(p):
    return 4.91 * (p**0.14 - (1-p)**0.14)

# Функция для построения графиков
def plot_graphs():
    N = int(N_entry.get())  # Получение значения из поля ввода для количества экспериментов
    M = int(M_entry.get())  # Получение значения из поля ввода для количества серий экспериментов
    ALPHA = float(alpha_entry.get())  # Получение значения из поля ввода для уровня доверия

    vs = exp_serial(M, N)  # Выполнение серии экспериментов

    confidence_interval = conf_interval(vs, ALPHA)  # Вычисление доверительного интервала

    mean_values = mean(vs)  # Вычисление средних значений

    exp_error = (confidence_interval[1,] - confidence_interval[0,]) / 2  # Вычисление экспериментальной ошибки

    coef = normal_quantile((1 + ALPHA) / 2)  # Вычисление коэффициента квантиля нормального распределения

    theory_error = np.zeros(N)  # Создание массива нулей размером N для теоретической ошибки
    for i in range(1, N + 1):
        theory_error[i-1] = coef * math.sqrt(0.5 * 0.5 / i)  # Расчет теоретической ошибки

    plt.figure(figsize=(10, 5))  # Создание фигуры для графиков

    # Построение графика экспериментальной ошибки
    plt.subplot(1, 2, 1)
    plt.xscale('log')
    for i in range(M):
        plt.plot(range(1, N+1), vs[i], color='black')  # Построение всех серий экспериментов
    plt.plot(range(1, N+1), confidence_interval[0,], color="blue")  # Построение нижней границы доверительного интервала
    plt.plot(range(1, N+1), confidence_interval[1,], color="blue")  # Построение верхней границы доверительного интервала
    plt.plot(range(1, N+1), mean_values, color="red")  # Построение графика средних значений
    plt.title('Экспериментальная ошибка')

    # Построение графика теоретической ошибки
    plt.subplot(1, 2, 2)
    plt.xscale('log')
    plt.plot(range(1, N+1), theory_error, color="blue")  # Построение графика теоретической ошибки
    plt.plot(range(1, N+1), exp_error, "r--")  # Построение графика экспериментальной ошибки
    plt.title('Теоретическая ошибка')
    plt.legend(['Теоретическая ошибка', 'Экспериментальная ошибка'])

    plt.tight_layout()  # Автоматическое размещение графиков

    # Вывод информации о среднем значении
    result_label.config(text=f'Среднее значение: {mean_values[-1]:.4f} +- {(confidence_interval[1,-1] - confidence_interval[0,-1]) / 2:.4f}')

    plt.show()  # Отображение графиков

# GUI
root = tk.Tk()  # Создание основного окна
root.title("Экспериментальная и теоретическая ошибки")  # Установка заголовка окна

main_frame = ttk.Frame(root)  # Создание главного фрейма
main_frame.grid(row=0, column=0, padx=10, pady=10)  # Размещение фрейма в окне

# Создание и размещение виджетов для ввода параметров эксперимента
N_label = ttk.Label(main_frame, text="Количество экспериментов:")
N_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
N_entry = ttk.Entry(main_frame)
N_entry.grid(row=0, column=1, padx=5, pady=5)

M_label = ttk.Label(main_frame, text="Количество серий экспериментов:")
M_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
M_entry = ttk.Entry(main_frame)
M_entry.grid(row=1, column=1, padx=5, pady=5)

alpha_label = ttk.Label(main_frame, text="Доверительный интервал:")
alpha_label.grid(row=2, column=0, padx=5, pady=5, sticky="e")
alpha_entry = ttk.Entry(main_frame)
alpha_entry.grid(row=2, column=1, padx=5, pady=5)

plot_button = ttk.Button(main_frame, text="Расчёт", command=plot_graphs)  # Создание кнопки для построения графиков
plot_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

result_label = ttk.Label(main_frame, text="")  # Метка для вывода результата
result_label.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

root.mainloop()  # Запуск цикла обработки событий окна