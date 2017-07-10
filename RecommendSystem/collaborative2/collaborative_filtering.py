#CF based on User Similarity, Eucedian Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as math

#column headers for the dataset
data_cols = ['user id', 'movie id', 'rating', 'timestamp']
item_cols = ['movie id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance ', 'Sci-Fi', 'Thriller', 'War', 'Western']
user_cols = ['user id', 'age', 'gender', 'occupation', 'zip code']

#importing the data files onto dataframes
users = pd.read_csv('/Users/carolinemeng/Desktop/ml-100k/u.user', sep='|', names=user_cols, encoding='latin-1')
item = pd.read_csv('/Users/carolinemeng/Desktop/ml-100k/u.item', sep='|', names=item_cols, encoding='latin-1')
data = pd.read_csv('/Users/carolinemeng/Desktop/ml-100k/u.data', sep='\t', names=data_cols, encoding='latin-1')

#sort the DataFrame by User ID and split the data-set into a training set and a test set
utrain = (data.sort_values('user id'))[:99832]
print(utrain.head())
print(utrain.tail())
utest = (data.sort_values('user id'))[99833:]
print(utest.head())
print ('***************************************************************\n')

#convert to a NumPy Array for ease of iteration
utrain = utrain.as_matrix(columns=['user id', 'movie id', 'rating'])
print(utrain)

# only interested in this user
utest = utest.as_matrix(columns=['user id', 'movie id', 'rating'])
print(utest)
print ('***************************************************************\n')

#a  list of users that contains a list of movies rated by the person
users_list = []
for i in range(1, 943):
    list = []
    for j in range(0, len(utrain)):
        if utrain[j][0] == i:
            list.append(utrain[j])
        else:
            break
    utrain = utrain[j:]
    users_list.append(list)

print 'There are %d many users with similarity' % (len(users_list))


def EucledianScore(train_user, test_user):
    sum = 0
    count = 0
    for i in test_user:
        score = 0
        for j in train_user:
            if(int(i[1]) == int(j[1])):
                score= ((float(i[2])-float(j[2]))*(float(i[2])-float(j[2])))
                count= count + 1
            sum = sum + score
    # if users have less than 4 movies in common then we assign them a high EucledianScore
    if(count<4):
        sum = 1000000
    return(math.sqrt(sum))

score_list = []
for i in range(0, len(users_list)):
    score_list.append([i + 1, EucledianScore(users_list[i], utest)])

score = pd.DataFrame(score_list, columns = ['user id','Eucledian Score'])
score = score.sort_values(by = 'Eucledian Score')
print(score)
score_matrix = score.as_matrix()


#Find the user with the lowest Euclidean score = highest similarity & get list of movies not watched
user= int(score_matrix[0][0])
common_list = []
full_list = []
for i in utest:
    for j in users_list[user-1]:
        if(int(i[1])== int(j[1])):
            common_list.append(int(j[1]))
        full_list.append(j[1])

common_list = set(common_list)
full_list = set(full_list)
recommendation = full_list.difference(common_list)

# group by movie titles, select the columns you need and then find the mean ratings of each movie
item_list = (((pd.merge(item,data).sort_values(by = 'movie id')).groupby('movie title')))['movie id', 'movie title', 'rating']
item_list = item_list.mean()
item_list['movie title'] = item_list.index
item_list = item_list.as_matrix()

# find the movies on item_list by IDs from recommendation. Then append them to a separate list
recommendation_list = []
for i in recommendation:
    recommendation_list.append(item_list[i - 1])

recommendation = (pd.DataFrame(recommendation_list, columns=['movie id', 'mean rating', 'movie title'])).sort_values(
    by='mean rating', ascending=False)
print(recommendation[['movie id', 'mean rating', 'movie title']])