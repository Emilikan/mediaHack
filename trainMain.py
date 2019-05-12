from pandas import read_csv, DataFrame
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import datasets
from sklearn.externals import joblib
import pickle
import pandas as pd

import numpy as np
from pylab import *

path = '0-2019-05-12(new).csv'

def newF():
    dataset = read_csv(path, ';')
    dataset.head()

    trg = dataset[['class']]
    trn = dataset.drop(['class'], axis=1)

    # разобьем данные на тестовую и обучающую выборку

    Xtrn, Xtest, Ytrn, Ytest = train_test_split(trn, trg, test_size=0.4)

    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(Xtrn, ravel(Ytrn))

    # сохранение модели
    joblib.dump(clf, 'filename.pkl')


def main():
    dataset = read_csv(path, ';')
    dataset.head()

    trg = dataset[['class']]
    trn = dataset.drop(['class'], axis=1)

    models = [LinearRegression(),  # метод наименьших квадратов
              RandomForestRegressor(n_estimators=100, max_features='sqrt'),  # случайный лес
              KNeighborsRegressor(n_neighbors=6),  # метод ближайших соседей
              SVR(kernel='linear'),  # метод опорных векторов с линейным ядром
              LogisticRegression()  # логистическая регрессия
              ]

    # разобьем данные на тестовую и обучающую выборку

    Xtrn, Xtest, Ytrn, Ytest = train_test_split(trn, trg, test_size=0.4)

    # создаем временные структуры
    TestModels = DataFrame()
    tmp = {}
    print('f')
    # для каждой модели из списка
    for model in models:
        # получаем имя модели
        m = str(model)
        tmp['Model'] = m[:m.index('(')]

        model.fit(Xtrn, ravel(Ytrn))
        tmp['R2_Y%s' % str(Ytrn.shape[1] + 1)] = r2_score(Ytest, model.predict(Xtest))
        # для каждого столбцам результирующего набора

        # записываем данные и итоговый DataFrame
        TestModels = TestModels.append([tmp])
    # делаем индекс по названию модели
    TestModels.set_index('Model', inplace=True)

    print('t')

    # отрисовка
    fig, axes = plt.subplots(ncols=2, figsize=(10, 4))
    TestModels.R2_Y1.plot(ax=axes[0], kind='bar', title='R2_Y1')
    TestModels.R2_Y2.plot(ax=axes[1], kind='bar', color='green', title='R2_Y2')



def mt():
    file = pd.read_csv('0-2019-05-11.csv', ';')
    Y = file['class']
    X = file[['holiday', 'weekends', 'time']]

    file2 = pd.read_csv('0-2019-05-12.csv', ';')
    Y1 = file2['class']
    X1 = file2[['holiday', 'weekends', 'time']]

    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, Y)
    print(clf.score(X1, Y1))



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

newF()