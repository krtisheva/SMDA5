from model import *
import numpy as np
import numpy.linalg as npl
import scipy.spatial as spp
import math


# Подпрограмма расчета определителя информационной матрицы
def indicator1(n, m, x):
    X = fill_obs_matrix(n, m, x)            # Заполнение матрицы наблюдений X
    X_t = np.transpose(X)                   # Транспонирование X
    X_tX = np.matmul(X_t, X)                # Произведение матриц X_t и X
    det_X = npl.det(X_tX / np.trace(X_tX))  # Нахождение определителя информационной матрицы деленной на ее след
    return det_X


# Подпрограмма поиска минимального собственного числа информационной матрицы
def indicator2(n, m, x):
    X = fill_obs_matrix(n, m, x)        # Заполнение матрицы наблюдений X
    X_t = np.transpose(X)               # Транспонирование X
    X_tX = np.matmul(X_t, X)            # Произведение матриц X_t и X
    l, v = npl.eigh(X_tX / np.trace(X_tX), UPLO='L')     # Нахождение собcтвенных чисел информационной матрицы
    l_min = np.amin(l)                  # Нахождение минимального собcтвенного числа информационной матрицы
    return l_min


# Подпрограмма расчета меры обусловленности матрицы по Нейману-Голдстейну
def indicator3(n, m, x):
    X = fill_obs_matrix(n, m, x)        # Заполнение матрицы наблюдений X
    X_t = np.transpose(X)               # Транспонирование X
    X_tX = np.matmul(X_t, X)            # Произведение матриц X_t и X
    l, v = npl.eigh(X_tX, UPLO='L')     # Нахождение собcтвенных чисел информационной матрицы
    l_min = np.amin(l)                  # Нахождение минимального собcтвенного числа информационной матрицы
    l_max = np.amax(l)                  # Нахождение максимального собcтвенного числа информационной матрицы
    return l_max / l_min


# Подпрограмма нахождения максимальной парной сопряженности
def indicator4(n, m, x):
    X = fill_obs_matrix(n, m, x)        # Заполнение матрицы наблюдений X
    X_t = np.transpose(X)               # Транспонирование X
    X_tX = np.matmul(X_t, X)            # Произведение матриц X_t и X
    R = np.zeros((m - 1, m - 1))        # Инициализация матрицы сопряженности
    max = 0.
    for i in range(0, m - 1):           # Заполнение матрицы сопряженности
        for j in range(0, m - 1):
            R[i][j] = 1 - spp.distance.cosine(X_tX[i], X_tX[j])
            if abs(R[i][j]) > max and i != j:
                max = abs(R[i][j])

    # Вывод на консоль матрицы R
    print(f"  \tx1\t\t\t\t\t\t\tx2\t\t\t\t\t\t\tx3\t\t\t\t\t\t\tx4\t\t\t\t\t\t\tx5\t\t\t\t\t\t\tx6\t\t\t\t\t\t\tx7")
    print(f"x1\t{R[0][0]}\t\t\t\t\t\t\t{R[0][1]}\t\t{R[0][2]}\t\t{R[0][3]}\t\t{R[0][4]}\t\t{R[0][5]}\t\t{R[0][6]}")
    print(f"x2\t{R[1][0]}\t\t{R[1][1]}\t\t\t\t\t\t\t{R[1][2]}\t\t{R[1][3]}\t\t{R[1][4]}\t\t{R[1][5]}\t\t{R[1][6]}")
    print(f"x3\t{R[2][0]}\t\t{R[2][1]}\t\t{R[2][2]}\t\t\t\t\t\t\t{R[2][3]}\t\t{R[2][4]}\t\t{R[2][5]}\t\t{R[2][6]}")
    print(f"x4\t{R[3][0]}\t\t{R[3][1]}\t\t{R[3][2]}\t\t{R[3][3]}\t\t\t\t\t\t\t{R[3][4]}\t\t{R[3][5]}\t\t{R[3][6]}")
    print(f"x5\t{R[4][0]}\t\t{R[4][1]}\t\t{R[4][2]}\t\t{R[4][3]}\t\t{R[4][4]}\t\t\t\t\t\t\t{R[4][5]}\t\t\t{R[4][6]}")
    print(f"x6\t{R[5][0]}\t\t{R[5][1]}\t\t{R[5][2]}\t\t{R[5][3]}\t\t{R[5][4]}\t\t\t{R[5][5]}\t\t\t\t\t\t\t{R[5][6]}")
    print(f"x7\t{R[6][0]}\t\t{R[6][1]}\t\t{R[6][2]}\t\t{R[6][3]}\t\t{R[6][4]}\t\t\t{R[6][5]}\t\t\t{R[6][6]}")

    return max


# Подпрограмма нахождения максимальной сопряженности
def indicator5(n, m, x):
    X = fill_obs_matrix(n, m, x)        # Заполнение матрицы наблюдений X
    X_t = np.transpose(X)               # Транспонирование X
    X_tX = np.matmul(X_t, X)            # Произведение матриц X_t и X
    R = np.zeros((m - 1, m - 1))        # Инициализация матрицы сопряженности
    max = 0.
    for i in range(0, m - 1):           # Заполнение матрицы сопряженности
        for j in range(0, m - 1):
            R[i][j] = 1 - spp.distance.cosine(X_tX[i], X_tX[j])
    R_1 = npl.inv(R)

    for i in range(0, m - 1):
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
