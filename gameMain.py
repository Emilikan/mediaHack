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

path = 'filename.pkl'
date = [[0, 1, 7]]


def main():
    # достаем модель с помощью пути
    clf = joblib.load(path)

    result = clf.predict_proba(date)
    print(result) # процент того, какой класс будет


main()
