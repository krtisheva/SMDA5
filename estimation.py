import numpy as np

from multicollinearity import *
import matplotlib.pyplot as plt


def ridge_estimation(n, m, x, y):
    X = fill_obs_matrix(n, m, x)    # Заполнение матрицы наблюдений X
    X_t = np.transpose(X)           # Транспонирование X
    X_tX = X_t @ X                  # Произведение матриц X_t и X
    diag = np.diag(X_tX)            # Диагональ матрицы X_t * X
    diag = np.diag(diag)
    k = 1000
    l = np.zeros(k)               # Массив значений параметра регуляризации
    theta = np.zeros((k, m))      # Массив значений оцененных параметров для различных параметров регуляризации

    for i in range(0, k):         # Нахождение оценок параметров при различных параметров регуляризации
        l[i] = i / 100000
        L = l[i] * diag
        theta[i] = npl.inv(X_tX + L) @ X_t @ y

    return theta, l


def pca(n, m, x, y):
    # Центрируем факторы и отклик
    avg_y = np.average(y)
    for i in range(m-1):
        avg_x = np.average(x[:, i])
        for j in range(n):
            x[j, i] -= avg_x
            y[j] -= avg_y

    # Заполняем матрицу наблюдений для центрированных факторов
    X = fill_obs_matrix(n, m, x)
    Xt_X = np.transpose(X) @ X

    # Нахождение собственных чисел и векторов матрицы
    l, v = npl.eigh(Xt_X, UPLO='L')

    Z = X @ v
    sum_l = l.sum()
    k = len(l)
    percent_l = np.zeros(k)
    for i in range(k):
        percent_l[i] = l[i] * 100.0 / sum_l
        print(f'lambda: {l[i]:.4f}\t\t percent: {percent_l[i]}')
    print('\n')

    i = k - 1
    while i != -1:
        # if percent_l[i] < 1:
        if i < k - 4:
            v = np.delete(v, i, 1)
            Z = np.delete(Z, i, 1)
            print(f'Removing {i} column')
        i -= 1
    print('\n')

    Zt = np.transpose(Z)
    b_pca = npl.inv(Zt @ Z)  @ Zt @ y
    theta_pca = v @ b_pca
    return theta_pca


def get_rss(theta, x, y, n, m):
    rss = 0
    X = fill_obs_matrix(n, m, x)
    y_est = X @ theta
    for i in range(n):
        rss += (y[i] - y_est[i])**2
    return rss

# Функция построение графика изменения остаточной суммы квадратов от параметра регуляризации
def plotting_RRS(n, m, x, y, theta, l):
    plt.Figure()
    plt.suptitle("График зависимости RSS от lambda")

    X = fill_obs_matrix(n, m, x)  # Заполнение матрицы наблюдений X
    k = len(l)
    RSS = np.zeros(k)
    for i in range(k):
        x_theta = X @ np.transpose(theta[i])
        RSS[i] = np.transpose(y - x_theta) @ (y - x_theta)

    plt.plot(l, RSS)
    plt.xlabel("lambda")
    plt.ylabel("RSS")
    plt.show()


# Функция построение графика изменения квадрата евклидовой нормы оценок параметров от параметра регуляризации
def plotting_norm(theta, l):
    plt.Figure()
    plt.suptitle("График зависимости ||theta|| от lambda")

    k = len(l)
    norm = np.zeros(k)
    for i in range(k):
        norm[i] = npl.norm(theta[i]) * npl.norm(theta[i])

    plt.plot(l, norm)
    plt.xlabel("lambda")
    plt.ylabel("||theta||")
    plt.show()