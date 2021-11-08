import numpy as np
from multicollinearity import *
import matplotlib.pyplot as plt


def ridge_estimation(n, m, x, y):
    X = fill_obs_matrix(n, m, x)    # Заполнение матрицы наблюдений X
    X_t = np.transpose(X)           # Транспонирование X
    X_tX = np.matmul(X_t, X)        # Произведение матриц X_t и X
    l = np.zeros(100)
    theta = np.zeros((100, m))

    for i in range(0, 100):
        l[i] = i / 10000
        L = l[i] * np.diag(X_tX)
        theta[i] = npl.inv(X_tX + L) @ X_t @ y

    return theta, l


# Функция построение графика
def plotting(n, m, x, y, theta, l):
    plt.Figure()
    plt.suptitle("График зависимости RSS от l")

    X = fill_obs_matrix(n, m, x)  # Заполнение матрицы наблюдений X
    RSS = np.zeros(100)
    for i in range(100):
        x_theta = X @ np.transpose(theta[i])
        RSS[i] = (y - np.transpose(x_theta)) @ (y - x_theta)

    plt.plot(l, RSS)
    plt.xlabel("l")
    plt.ylabel("RSS")
    plt.show()
