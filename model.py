# Функция вычисления компонент вектора f для конкретной точки
def f(x):
    return [1, x[0], x[1], x[2], x[3], x[4], x[5], x[6]]


# Функция вычисления компонент вектора f для конкретной точки (метод главных компонент)
def f_pca(x):
    return [x[0], x[1], x[2], x[3], x[4], x[5], x[6]]


# Функция вичисления значения модели объекта, для конкретных
# точек плана и вектора параметров
def model(x, theta):
    f_x = f(x)
    etta = 0.
    for i in range(0, 8):
        etta += theta[i] * f_x[i]
    return etta
