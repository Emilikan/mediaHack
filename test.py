vkapi = pd.read_csv(filename, sep=';', low_memory=False)
vkapi.head(10)
vkapi.columns


def likes():
    fig, ax = plt.subplots(1, 1)
    ax.plot(vkapi['year'], vkapi['likes'])
    fig.set_size_inches(22, 10)
    plt.savefig('likes.png')
    plt.title('Статистика по максимальному значению лайков за год', fontsize=18)

def reposts():
    fig, ax = plt.subplots(1, 1)
    ax.plot(vkapi['year'], vkapi['reposts'])
    fig.set_size_inches(20, 10)
    plt.savefig('reposts.png')
    plt.title('Статистика по максимальному значению репостов за год', fontsize=14)



def likes_and_posts():
    %matplotlib inline
    plt.style.use('bmh')

    likes = vkapi.likes
    plt.figure(num=5, figsize=(20, 7))
    plt.title('Количество записей на стене и количество лайков')
    plt.xlabel('count')
    plt.ylabel('likes')
    plt.plot(likes, '-')
    plt.savefig('likes_and_posts.png')


def most_reposts():
    weekday_reposts_summary = vkapi.groupby(['weekday']).reposts.mean()[['Monday', time_reposts_summary = vkapi.groupby(['time']).reposts.mean()
    plt.figure(num=1, figsize=(30, 15))
    plt.style.use('classic')
    plt.plot(time_reposts_summary, '-')
    plt.savefig('most_reposts.png')



def time_of_posts():
    time_summary = vkapi.time.value_counts()[
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]]
    time_summary = vkapi.time.value_counts().sort_index()
    plt.figure(num=1, figsize=(20, 8))
    time_summary.plot.barh(stacked=True, alpha=0.7)
    plt.style.use('fivethirtyeight')
    plt.savefig('time_of_reposts.png')