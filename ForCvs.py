import pandas as pd
import csv
import datetime
import numpy as np


def main():
    file = pd.read_csv('30666517-2019-05-11.csv', ';')
    # file = file.dropna(axis='index', how='any', subset=['text'])


    date = file['date']
    day = file['weekday']
    time = file['time']
    likes = file['likes']
    reposts = file['reposts']
    views = file['views']
    text = file[['text']]
    link = file[['link']]

    # определяем среднии значения
    max_laik = file['likes'].describe()
    max_rep = file['reposts'].describe()
    max_views = file['views'].describe()

    mean_likes = max_laik[['mean'][0]]
    mean_rep = max_rep[['mean'][0]]
    mean_views = max_views[['mean'][0]]

    c = -1
    print(date[2000][:5])
    filtered_data = []
    for i in file['id']:
        c += 1
        # print(c, i)
        numberOfClass_end = one(likes[c], reposts[c], views[c], mean_likes, mean_rep, mean_views)
        time_end = timeF(time[c])
        day_end = dayF(day[c])
        date_end = dateF(date[c][:5])

        filtered_post = {
            'id': c,
            'class': numberOfClass_end,
            'holiday': date_end,
            'weekends': day_end,
            'time': time_end,
        }

        filtered_data.append(filtered_post)

    write_csv(filtered_data)


def one(likes, reposts, views, mean_likes, mean_reposts, mean_views):
    if ((likes > mean_likes and reposts > mean_reposts) or (likes > mean_likes and views > mean_views)
            or (reposts > mean_reposts and views > mean_views)
            or (likes > mean_likes and reposts > mean_reposts and views > mean_views)):
        return 1
    else:
        return 0


def timeF(time):
    if time >= 0 and time < 3:
        return 0
    elif time >= 3 and time < 6:
        return 1
    elif time >= 6 and time < 9:
        return 2
    elif time >= 9 and time < 12:
        return 3
    elif time >= 12 and time < 15:
        return 4
    elif time >= 15 and time < 18:
        return 5
    elif time >= 18 and time < 21:
        return 6
    else:
        return 7


def dayF(day):
    if day == 'Saturday' or day == 'Sunday':
        return 0
    else:
        return 1


def dateF(date):
    if date == '01-01' or date == '31-12' or date == '23-02' or date == '08-03':
        return 1
    else:
        return 0


def write_csv(data, encoding='utf-8'):
    owner_id = data[0]['id']
    filename = '{owner_id}-{datetime}.csv'.format(owner_id=owner_id, datetime=str(datetime.datetime.now())[:10])
    with open(filename, 'w', newline='', encoding=encoding) as csvfile:
        fieldnames = ['id', 'class', 'holiday', 'weekends', 'time']

        writer = csv.DictWriter(csvfile, delimiter=';', fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(data)

        print('Data written to csv', filename)
    csvfile.close()


main()
