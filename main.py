from gen_data import *
from estimation import *

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
l_min, v_min = indicator2(n, m, x)
print(f"Минимальное собственное число информационной матрицы = {l_min}")
print(f"Собственный вектор соответсвующий мин. собств. числу: {v_min}\n")

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

# Ридж-оценивание
theta, reg_params = ridge_estimation(n, m, x, y)
plotting_rss(n, m, x, y, theta, reg_params)
plotting_norm(theta, reg_params)

# Выбор параметра регуляции для оценивания
estims = {reg_params[i]: theta[i] for i in range(len(reg_params))}
reg_p = float(input('Выберите параметр регуляции для получения оценки (>0 c шагом 0.00001):\n'))
if reg_p >= 0:
    print(f'Параметр регуляции = {reg_p}')
    print(f'Ридж-оценка: {estims[reg_p]}')
    rel_err = npl.norm(np.array([1, 1, 1, 1, 1, 1, 1, 1]) - estims[reg_p])
    print(f'Норма разности: {rel_err}')
    rss = get_rss(estims[reg_p], x, y, n, m)
    print(f'Остаточная вариация: {rss}\n\n')
    plotting_y(x, y, n, m, estims[reg_p], 'Ридж-оценивание')

# Оценивание по методы главных компонент
x_c = np.array(x)
y_c = np.array(y)
theta_pca = pca(n, m, x_c, y_c)
print(f'Оценка по методу главных компонент: {theta_pca}')
rel_err = npl.norm(np.array([1, 1, 1, 1, 1, 1, 1, 1]) - theta_pca)
print(f'Норма разности: {rel_err}')
rss = get_rss(theta_pca, x, y, n, m)
print(f'Остаточная вариация: {rss}\n\n')
plotting_y(x, y, n, m, theta_pca, 'Метод главных компонент')
