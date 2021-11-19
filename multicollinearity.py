from model import *
import numpy as np
import numpy.linalg as npl
import scipy.spatial as spp
import math


# Подпрограмма расчета определителя информационной матрицы
def indicator1(n, m, x):
    xt_x = x_tx(n, m, x)                    # Произведение матриц X_t и X
    det_x = npl.det(xt_x / np.trace(xt_x))  # Нахождение определителя информационной матрицы деленной на ее след
    return det_x


# Подпрограмма поиска минимального собственного числа информационной матрицы
def indicator2(n, m, x):
    xt_x = x_tx(n, m, x)                                    # Произведение матриц X_t и X
    l, v = npl.eigh(xt_x / np.trace(xt_x), UPLO='L')        # Нахождение собcтвенных чисел информационной матрицы
    l_min = np.amin(l)                                      # Нахождение минимального собcтвенного числа XtX/tr(XtX)
    v_min = get_eigh_vector(m, l, l_min, v)
    return l_min, v_min


# Подпрограмма выбора собственного вектора соотв. собственному числу
def get_eigh_vector(m, lambdas, l_min, v):
    for i in range(m - 1):
        if lambdas[i] == l_min:
            return v[:, i]


# Подпрограмма расчета меры обусловленности матрицы по Нейману-Голдстейну
def indicator3(n, m, x):
    xt_x = x_tx(n, m, x)                    # Произведение матриц X_t и X
    l, v = npl.eigh(xt_x, UPLO='L')         # Нахождение собcтвенных чисел информационной матрицы
    l_min = np.amin(l)                      # Нахождение минимального собcтвенного числа информационной матрицы
    l_max = np.amax(l)                      # Нахождение максимального собcтвенного числа информационной матрицы
    return l_max / l_min


# Подпрограмма нахождения максимальной парной сопряженности
def indicator4(n, m, x):
    xt_x = x_tx(n, m, x)                # Произведение матриц X_t и X
    r = np.zeros((m-1, m-1))            # Инициализация матрицы сопряженности
    max = 0.                            # Максимальная парная сопряженность
    for i in range(0, m-1):             # Заполнение матрицы сопряженности R
        for j in range(0, m-1):
            r[i][j] = 1 - spp.distance.cosine(xt_x[i+1], xt_x[j+1])
            if abs(r[i][j]) > max and i != j:
                max = abs(r[i][j])

    print("Матрица R:")
    output_matrix(r)                        # Вывод на консоль матрицы R
    return max


# Подпрограмма нахождения максимальной сопряженности
def indicator5(n, m, x):
    xt_x = x_tx(n, m, x)                    # Произведение матриц X_t и X
    r = np.zeros((m-1, m-1))                    # Инициализация матрицы сопряженности

    for i in range(0, m-1):                   # Заполнение матрицы сопряженности
        for j in range(0, m-1):
            r[i][j] = 1 - spp.distance.cosine(xt_x[i+1], xt_x[j+1])

    r_1 = npl.inv(r)                        # Нахождение матрицы, обратной к матрице сопряженности R

    print("Матрица R_1:")
    output_matrix(r_1)                      # Вывод на консоль матрицы R_1
    print()

    max = 0.                                # Максимальная сопряженность
    num = -1                                # Номер фактора
    for i in range(0, m-1):                   # Поиск максимальной сопряженности
        ri_2 = 1 - 1 / r_1[i][i]
        ri = abs(math.sqrt(ri_2))
        print(f"R{i+1} = {ri}")
        if ri > max:
            max = ri
            num = i+1

    return max, num


# Функция заполнения матрицы наблюдений X
def fill_obs_matrix(n, m, x):
    mat_x = np.zeros((n, m))  # x - матрица наблюдений
    # В цикле от 0 до n вычисляем компоненты вектора f
    # для каждого наблюдения и заносим в матрицу X, как
    # новую строку, итого n строк по m элементов (n*m)
    for i in range(0, n):
        mat_x[i] = f(x[i])
    return mat_x


# Функция заполнения матрицы наблюдений X
def fill_obs_matrix_pca(n, m, x):
    mat_x = np.zeros((n, m-1))  # x - матрица наблюдений
    # В цикле от 0 до n вычисляем компоненты вектора f
    # для каждого наблюдения и заносим в матрицу X, как
    # новую строку, итого n строк по m элементов (n*m)
    for i in range(0, n):
        mat_x[i] = f_pca(x[i])
    return mat_x


# Функция нахождения произведения матриц X_t и X
def x_tx(n, m, x):
    x = fill_obs_matrix(n, m, x)            # Заполнение матрицы наблюдений X
    xt = np.transpose(x)                   # Транспонирование X
    xt_x = np.matmul(xt, x)                # Произведение матриц X_t и X
    return xt_x


# Функция вывода матрицы на консоль
def output_matrix(m):
    print(f"  \tx1\t\t\t\tx2\t\t\t\tx3\t\t\t\tx4\t\t\t\tx5\t\t\t\tx6\t\t\t\tx7")
    for i in range(7):
        print(f'x{i+1}\t{m[i, 0]:.6e}\t{m[i, 1]:.6e}\t{m[i, 2]:.6e}\t{m[i, 3]:.6e}\t'
              f'{m[i, 4]:.6e}\t{m[i, 5]:.6e}\t{m[i, 6]:.6e}')
