from multicollinearity import *
import matplotlib.pyplot as plt


def ridge_estimation(n, m, x, y):
    x = fill_obs_matrix(n, m, x)        # Заполнение матрицы наблюдений X
    xt = np.transpose(x)                # Транспонирование X
    xt_x = xt @ x                       # Произведение матриц X_t и X
    diag = np.diag(xt_x)                # Диагональ матрицы X_t * X
    diag = np.diag(diag)
    k = 1000
    reg_params = np.zeros(k)            # Массив значений параметра регуляризации
    theta = np.zeros((k, m))            # Массив значений оцененных параметров для различных параметров регуляризации

    for i in range(0, k):               # Нахождение оценок параметров для различных параметров регуляризации
        reg_params[i] = i / 100000
        params_matrix = reg_params[i] * diag
        theta[i] = npl.inv(xt_x + params_matrix) @ xt @ y

    return theta, reg_params


# Подпрограмма нахождения оценок с помощью метода главных компонент
def pca(n, m, x, y):
    # Центрируем факторы и отклик
    avg_y = np.average(y)
    avg = np.zeros(m-1)
    for i in range(m-1):
        avg_x = np.average(x[:, i])
        avg[i] = avg_x
        for j in range(n):
            x[j, i] -= avg_x
            y[j] -= avg_y

    # Заполняем матрицу наблюдений для центрированных факторов
    x_obs = fill_obs_matrix_pca(n, m, x)
    xt_x = np.transpose(x_obs) @ x_obs

    # Нахождение собственных чисел и векторов матрицы
    l, v = npl.eigh(xt_x, UPLO='L')

    z = x_obs @ v                                               # Нахождение матрицы значений главных компонент

    # Вычисление влияния каждой компоненты на изменчивость центрированных факторов
    sum_l = l.sum()
    k = len(l)
    percent_l = np.zeros(k)
    for i in range(k):
        percent_l[i] = l[i] * 100.0 / sum_l
        print(f'lambda: {l[i]:.4f}\t\t percent: {percent_l[i]:.4f}')
    print('\n')

    # Исключение компонент менее всего влияющих на изменчивость
    i = k - 1
    while i != -1:
        if percent_l[i] < 1:
            v = np.delete(v, i, 1)
            z = np.delete(z, i, 1)
            print(f'Removing {i} column')
        i -= 1
    print('\n')

    # Вычисление оценок B для регрессии главных компонент
    zt = np.transpose(z)
    b_pca = npl.inv(zt @ z)  @ zt @ y
    theta_pca = v @ b_pca
    theta = np.zeros(m)
    theta[0] = avg_y - np.transpose(avg) @ theta_pca
    theta[1:] = theta_pca
    return theta

# Подпрограмма вычисления остаточной вариации
def get_rss(theta, x, y, n, m):
    rss = 0
    x_obs = fill_obs_matrix(n, m, x)
    y_est = x_obs @ theta
    for i in range(n):
        rss += (y[i] - y_est[i])**2
    return rss


# Функция построение графика изменения остаточной суммы квадратов от параметра регуляризации
def plotting_rss(n, m, x, y, theta, reg_params):
    plt.Figure()
    plt.suptitle("График зависимости RSS от lambda")

    x_obs = fill_obs_matrix(n, m, x)  # Заполнение матрицы наблюдений X
    k = len(reg_params)
    rss = np.zeros(k)
    for i in range(k):
        x_theta = x_obs @ np.transpose(theta[i])
        rss[i] = np.transpose(y - x_theta) @ (y - x_theta)

    plt.plot(reg_params, rss)
    plt.xlabel("lambda")
    plt.ylabel("RSS")
    plt.show()


# Функция построение графика изменения квадрата евклидовой нормы оценок параметров от параметра регуляризации
def plotting_norm(theta, reg_params):
    plt.Figure()
    plt.suptitle("График зависимости ||theta|| от lambda")

    k = len(reg_params)
    norm = np.zeros(k)
    for i in range(k):
        norm[i] = npl.norm(theta[i]) * npl.norm(theta[i])

    plt.plot(reg_params, norm)
    plt.xlabel("lambda")
    plt.ylabel("||theta||")
    plt.show()


# Построение графиков значений откликов
def plotting_y(x, y, n, m, theta, method):
    plt.Figure()
    plt.suptitle("График y и y_est")
    plt.title(method)

    t = np.array([a for a in range(n)])
    x_obs = fill_obs_matrix(n, m, x)
    y_est = x_obs @ theta

    plt.plot(t, y)
    plt.plot(t, y_est)
    plt.legend(['y', 'y предсказанное'])
    plt.show()
