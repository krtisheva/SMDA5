import random
import math
import matplotlib.pyplot as plt
import numpy as np
from model import *


# Подпрограмма генерации экспериментальных данных
def data_gen():
    n = 1000                         # количество измерений
    m = 8                           # количество параметров модели со свободным членом
    etta = np.empty(n)              # массив значений незашумленного отклика
    y = np.empty(n)                 # массив значений зашумленного отклика
    x = np.empty((n, m-1))     # массив значений каждого фактора
    theta = [1, 1, 1, 1, 1, 1, 1, 1]     # истинные значения оцениваемых параметров

    # заполнение векторов etta, x1, x2
    for i in range(n):
        x1 = random.uniform(-1, 1)
        x2 = random.uniform(-1, 1)
        x3 = random.uniform(-1, 1)
        x4 = random.uniform(-1, 1)
        x5 = random.uniform(-1, 1)
        x6 = random.uniform(-1, 1)
        x[i] = [x1, x2, x3, x4, x5, x6, x4+x5+x6+random.normalvariate(0, 0.01)]
        etta[i] = model(x[i], theta)
        i += 1

    avg = sum(etta) / n     # среднее значение незашумленного отклика
    # вычисление мощности сигнала
    omega2 = sum(list((etta[i] - avg) ** 2 for i in range(0, n))) / (n - 1)
    variation = 0.1 * omega2

    for i in range(0, n):
        e = random.normalvariate(0, math.sqrt(variation))
        y[i] = etta[i] + e

    return n, m, x, y


# Функция вывода данных в файл унифицированной структуры
def output_data(n, m, x, y):
    # Открытие файла
    f_out = open("modelled_data.txt", "w")
    f_out.write("%d\t%d\n" % (n, m))
    for i in range(0, n):
        for j in range(0, 7):
            f_out.write("%f\t" % x[i][j])
        f_out.write("%f\n" % y[i])
    f_out.close()
    print("Сгенерированные данные вывели в файл 'modelled_data.txt'!")


# Функция ввода данных о выборке измерений
def input_data():
    with open('modelled_data.txt', 'r') as f_in:
        s = f_in.readline().split()
        n = int(s[0])    # n - число измерений
        m = int(s[1])    # m - число параметров
        y = np.empty(n)  # массив значений зашумленного отклика
        x = np.empty((n, m-1))  # массив значений каждого фактора

        # цикл по всем строкам файла и заполнение определенных выше массивов
        i = 0
        for line in f_in:
            s = line.split()
            x[i] = [float(s[0]), float(s[1]), float(s[2]), float(s[3]), float(s[4]), float(s[5]), float(s[6])]
            y[i] = float(s[7])
            i += 1
    return n, m, x, y
