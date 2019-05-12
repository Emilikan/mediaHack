import csv
import datetime
import requests
import time
import datetime
import inline
import pandas as pd
import matplotlib.pyplot as plt
import csv

access_token = 'qTJNHDXpPM488eEBvbHr'
owner_id = '-101965347'

vkapi = pd.read_csv('30666517-2019-05-11.csv', sep=';', low_memory=False)
vkapi.head(10)
vkapi.columns


def main():
    file = pd.read_csv('30666517-2019-05-11.csv', ';')

    # определяем среднии значения
    max_laik = file['likes'].describe()
    max_rep = file['reposts'].describe()
    max_views = file['views'].describe()

    mean_likes_end = max_laik[['mean'][0]]
    mean_rep_end = max_rep[['mean'][0]]
    mean_views_end = max_views[['mean'][0]]

    max_likes_end = max_laik[['max'][0]]
    max_rep_end = max_rep[['max'][0]]
    max_views_end = max_views[['max'][0]]

    min_likes_end = max_laik[['min'][0]]
    min_rep_end = max_rep[['min'][0]]
    min_views_end = max_views[['min'][0]]

    likes()
    reposts()

    return ('Средние значения:\nЛайков:' + str(mean_likes_end) + '\nРепостов:' + str(mean_rep_end) + '\nПросмотрв:' + str(mean_views_end) + '\nМинимальные значения:\nЛайков:' + str(min_likes_end) + '\nРепостов:' + str(min_rep_end) + '\nПросмотров:' + str(min_views_end) + '\nМаксимальные значния:\nЛайков:' + str(max_likes_end) + '\nРепостов:' + str(max_rep_end) + '\nПросмотров:' + str(max_views_end))


def likes():
    fig, ax = plt.subplots(1, 1)
    ax.plot(vkapi['year'], vkapi['likes'])
    fig.set_size_inches(22, 10)
    plt.title('Статистика по максимальному значению лайков за год', fontsize=18)
    plt.savefig('likes.png')
    print('ok')


def reposts():
    fig, ax = plt.subplots(1, 1)
    ax.plot(vkapi['year'], vkapi['reposts'])
    fig.set_size_inches(20, 10)

    plt.title('Статистика по максимальному значению репостов за год', fontsize=14)
    plt.savefig('reposts.png')

