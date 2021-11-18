from model import *
import numpy as np
import numpy.linalg as npl
import scipy.spatial as spp
import math


# Подпрограмма расчета определителя информационной матрицы
def indicator1(n, m, x):
    X_tX = x_tx(n, m, x)                    # Произведение матриц X_t и X
    det_X = npl.det(X_tX / np.trace(X_tX))  # Нахождение определителя информационной матрицы деленной на ее след
    return det_X


# Подпрограмма поиска минимального собственного числа информационной матрицы
def indicator2(n, m, x):
    X_tX = x_tx(n, m, x)                                    # Произведение матриц X_t и X
    l, v = npl.eigh(X_tX / np.trace(X_tX), UPLO='L')        # Нахождение собcтвенных чисел информационной матрицы
    l_min = np.amin(l)                                      # Нахождение минимального собcтвенного числа информационной матрицы
    v_min = get_eigh_vector(m, l, l_min, v)
    return l_min, v_min


def get_eigh_vector(m, l, l_min, v):
    for i in range(m):
        if l[i] == l_min:
            return v[:, i]


# Подпрограмма расчета меры обусловленности матрицы по Нейману-Голдстейну
def indicator3(n, m, x):
    X_tX = x_tx(n, m, x)                    # Произведение матриц X_t и X
    l, v = npl.eigh(X_tX, UPLO='L')         # Нахождение собcтвенных чисел информационной матрицы
    l_min = np.amin(l)                      # Нахождение минимального собcтвенного числа информационной матрицы
    l_max = np.amax(l)                      # Нахождение максимального собcтвенного числа информационной матрицы
    return l_max / l_min


# Подпрограмма нахождения максимальной парной сопряженности
def indicator4(n, m, x):
    X_tX = x_tx(n, m, x)                    # Произведение матриц X_t и X
    R = np.zeros((m - 1, m - 1))            # Инициализация матрицы сопряженности
    max = 0.                                # Максимальная парная сопряженность
    for i in range(1, m):                   # Заполнение матрицы сопряженности R
        for j in range(1, m):
            R[i-1][j-1] = 1 - spp.distance.cosine(X_tX[i], X_tX[j])
            if abs(R[i-1][j-1]) > max and i != j:
                max = abs(R[i-1][j-1])

    print("Матрица R:")
    output_matrix(R)                        # Вывод на консоль матрицы R
    return max


# Подпрограмма нахождения максимальной сопряженности
def indicator5(n, m, x):
    X_tX = x_tx(n, m, x)                    # Произведение матриц X_t и X
    R = np.zeros((m - 1, m - 1))            # Инициализация матрицы сопряженности

    for i in range(1, m):                   # Заполнение матрицы сопряженности
        for j in range(1, m):
            R[i-1][j-1] = 1 - spp.distance.cosine(X_tX[i], X_tX[j])

    R_1 = npl.inv(R)                        # Нахождение матрицы, обратной к матрице сопряженности R

    print("Матрица R_1:")
    output_matrix(R_1)                      # Вывод на консоль матрицы R_1
    print()

    max = 0.                                # Максимальная сопряженность
    num = -1                                # Номер фактора
    for i in range(0, m - 1):               # Поиск максимальной сопряженности
        Ri_2 = 1 - 1 / R_1[i][i]
        Ri = abs(math.sqrt(Ri_2))
        print(f"R{i+1} = {Ri}")
        if Ri > max:
            max = Ri
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


# Функция нахождения произведения матриц X_t и X
def x_tx(n, m, x):
    X = fill_obs_matrix(n, m, x)            # Заполнение матрицы наблюдений X
    X_t = np.transpose(X)                   # Транспонирование X
    X_tX = np.matmul(X_t, X)                # Произведение матриц X_t и X
    return X_tX


# Функция вывода матрицы на консоль
def output_matrix(m):
    print(f"  \tx1\t\t\t\tx2\t\t\t\tx3\t\t\t\tx4\t\t\t\tx5\t\t\t\tx6\t\t\t\tx7")
    print("x1\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e" % (m[0][0], m[0][1], m[0][2], m[0][3], m[0][4], m[0][5], m[0][6]))
    print("x2\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e" % (m[1][0], m[1][1], m[1][2], m[1][3], m[1][4], m[1][5], m[1][6]))
    print("x3\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e" % (m[2][0], m[2][1], m[2][2], m[2][3], m[2][4], m[2][5], m[2][6]))
    print("x4\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e" % (m[3][0], m[3][1], m[3][2], m[3][3], m[3][4], m[3][5], m[3][6]))
    print("x5\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e" % (m[4][0], m[4][1], m[4][2], m[4][3], m[4][4], m[4][5], m[4][6]))
    print("x6\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e" % (m[5][0], m[5][1], m[5][2], m[5][3], m[5][4], m[5][5], m[5][6]))
    print("x7\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e" % (m[6][0], m[6][1], m[6][2], m[6][3], m[6][4], m[6][5], m[6][6]))
