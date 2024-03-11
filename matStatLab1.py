from random import randint
import numpy as np
import matplotlib.pyplot as plt
import math
import tkinter as tk
from tkinter import ttk

# Функция для моделирования броска монеты
def flip_coin():
    return randint(0, 1)

# Функция для выполнения одного эксперимента (бросания монеты)
def experiment(N):
    result = np.zeros(N)
    counter = 0
    for i in range(N):
        fc = flip_coin()
        if fc == 1:
            counter += 1
        result[i] = counter / (i + 1)
    return result

# Функция для выполнения серии экспериментов
def exp_serial(M, N):
    result = np.zeros((M, N))
    for i in range(M):
        result[i,] = experiment(N)
    return result

# Функция для расчета среднего значения массива
def mean(vs):
    return np.mean(vs, axis=0)

# Функция для вычисления доверительного интервала
def conf_interval(vs, alpha):
    m = vs.shape[0]
    a = (1 - alpha) / 2
    m_down = int(m * a)
    m_up = m - m_down - 1

    sorted_vs = np.sort(vs, axis=0)

    return np.apply_along_axis(lambda x: np.array([x[m_down], x[m_up]]), 0, sorted_vs)

# Функция для вычисления квантиля нормального распределения
def normal_quantile(p):
    return 4.91 * (p**0.14 - (1-p)**0.14)

# Функция для построения графиков
def plot_graphs():
    N = int(N_entry.get())
    M = int(M_entry.get())
    ALPHA = float(alpha_entry.get())

    vs = exp_serial(M, N)

    confidence_interval = conf_interval(vs, ALPHA)

    mean_values = mean(vs)

    exp_error = (confidence_interval[1,] - confidence_interval[0,]) / 2

    coef = normal_quantile((1 + ALPHA) / 2)

    theory_error = np.zeros(N)
    for i in range(1, N + 1):
        theory_error[i-1] = coef * math.sqrt(0.5 * 0.5 / i)

    plt.figure(figsize=(10, 5))

    # Построение графика экспериментальной ошибки
    plt.subplot(1, 2, 1)
    plt.xscale('log')
    for i in range(M):
        plt.plot(range(1, N+1), vs[i], color='black')
    plt.plot(range(1, N+1), confidence_interval[0,], color="blue")
    plt.plot(range(1, N+1), confidence_interval[1,], color="blue")
    plt.plot(range(1, N+1), mean_values, color="red")
    plt.title('Экспериментальная ошибка')

    # Построение графика теоретической ошибки
    plt.subplot(1, 2, 2)
    plt.xscale('log')
    plt.plot(range(1, N+1), theory_error, color="blue")
    plt.plot(range(1, N+1), exp_error, "r--")
    plt.title('Теоретическая ошибка')
    plt.legend(['Теоретическая ошибка', 'Экспериментальная ошибка'])

    plt.tight_layout()

    # Вывод информации о среднем значении
    result_label.config(text=f'Среднее значение: {mean_values[-1]:.4f} +- {(confidence_interval[1,-1] - confidence_interval[0,-1]) / 2:.4f}')

    plt.show()

# GUI
root = tk.Tk()
root.title("Экспериментальная и теоретическая ошибки")

main_frame = ttk.Frame(root)
main_frame.grid(row=0, column=0, padx=10, pady=10)

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

plot_button = ttk.Button(main_frame, text="Расчёт", command=plot_graphs)
plot_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

result_label = ttk.Label(main_frame, text="")
result_label.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

root.mainloop()