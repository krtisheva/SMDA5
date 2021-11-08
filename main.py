from gen_data import *
from multicollinearity import *
from estimation import  *

answer = input('Хотите сгенерировать новые данные? (y/n)\n')
if answer == 'y':
    n, m, x, y = data_gen()
    output_data(n, m, x, y)

n, m, x, y = input_data()

# Расчет первого показателя мультиколлинеарности
print("Показатель мультиколлинеарности №1")
det = indicator1(n, m, x)
print(f"Определитель информационной матрицы = {det}\n")

# Расчет второго показателя мультиколлинеарности
print("Показатель мультиколлинеарности №2")
l_min = indicator2(n, m, x)
print(f"Минимальное собственное число информационной матрицы = {l_min}\n")

# Расчет третьего показателя мультиколлинеарности
print("Показатель мультиколлинеарности №3")
l_max_min = indicator3(n, m, x)
print(f"Мера обусловленности матрицы по Нейману-Голдстейну = {l_max_min}\n")

# Расчет четвертого показателя мультиколлинеарности
print("Показатель мультиколлинеарности №4")
max_pair = indicator4(n, m, x)
print(f"Максимальная парная сопряженность = {max_pair}\n")

# Расчет пятого показателя мультиколлинеарности
print("Показатель мультиколлинеарности №5")
max, num = indicator5(n, m, x)
print(f"Максимальная сопряженность = {max} (i={num})\n")

theta, l = ridge_estimation(n, m, x, y)
plotting(n, m, x, y, theta, l)
