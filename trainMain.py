from pandas import read_csv, DataFrame
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import numpy as np
from pylab import *

path = 'EnergyEfficiency/ENB2012_data.csv'


def main():
    dataset = read_csv(path, ';')
    dataset.head()

    models = [LinearRegression(),  # метод наименьших квадратов
              RandomForestRegressor(n_estimators=100, max_features='sqrt'),  # случайный лес
              KNeighborsRegressor(n_neighbors=6),  # метод ближайших соседей
              SVR(kernel='linear'),  # метод опорных векторов с линейным ядром
              LogisticRegression()  # логистическая регрессия
              ]

    # разобьем данные на тестовую и обучающую выборку


def log_regr():
    # Для вопроизодимости результатов, зависящих от генератора случайных чисел
    np.random.seed(1000)

    # Настройка шрифтов для будущих графиков
    rcParams['font.family'] = 'DejaVu Sans'  # Понимает русские буквы
    rcParams['font.size'] = 16

    # данные 1 и 2 классов
    data1=''
    data2=''

    # далее получить x и y
    X = np.vstack([data1, data2])
    Y = [0]*len(data1) + [1]*len(data2)
    #

    # Настраиваем модель логистической регрессии
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X, Y)

    # Далее получаем массивы для классификации
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    #

    # Выполнение классификации каждой точки массива
    Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

    # логарифмы вероятностей принадлежности к классам
    probabilities = logreg.predict_log_proba(np.c_[xx.ravel(), yy.ravel()])

    # преобразование формата (как требует pcolormesh)
    Z = Z.reshape(xx.shape)
    p1 = probabilities[:, 0].reshape(xx.shape)
    p2 = probabilities[:, 1].reshape(xx.shape)


    # Отрисовка результатов классификации и оценок вероятностей

    title(u'Результаты классификации')
    plot(data1[:, 0], data1[:, 1], 'oy')
    plot(data2[:, 0], data2[:, 1], 'sc')
    pcolormesh(xx, yy, Z, cmap='RdYlBu')
    gca().set_xlim([x_min, x_max])
    gca().set_ylim([y_min, y_max])

    figure()
    title(u'Лог-Вероятности класса 1')
    pcolormesh(xx, yy, p1, cmap='cool')
    colorbar()
    gca().set_xlim([x_min, x_max])
    gca().set_ylim([y_min, y_max])

    figure()
    title(u'Лог-Вероятности класса 2')
    pcolormesh(xx, yy, p2, cmap='cool')
    colorbar()
    gca().set_xlim([x_min, x_max])
    gca().set_ylim([y_min, y_max])

    show()
