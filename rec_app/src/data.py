import pandas as pd
import datetime, time
import os
import random
import numpy as np
import scipy.sparse as sp
import json
from IPython.display import Image
import base64
from imdbpie import Imdb
import requests

DATA_DIR = "../../movielens-imdb-exploration/data"

def string2ts(string, fmt="%Y-%m-%d %H:%M:%S"):
    dt = datetime.datetime.strptime(string, fmt)
    t_tuple = dt.timetuple()
    return int(time.mktime(t_tuple))

def slice_by_lengths(lengths, the_list):
    for length in lengths:
        new = []
        for i in range(length):
            new.append(the_list.pop(0))
        yield new


def initial_data():
    # MOVIES
    df_movies = pd.read_csv(f"{DATA_DIR}/movies_cast_company.csv", encoding='utf8')
    df_movies["cast"] = df_movies["cast"].apply(lambda x: json.loads(x))
    df_movies["company"] = df_movies["company"].apply(lambda x: json.loads(x))

    # TODO: just temporary, later remove
    df_movies = df_movies.drop(['movie_id', 'keyword', 'cast', 'company'], axis=1)


    # RATINGS
    df_ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")
    df_ratings.rating_timestamp = df_ratings.rating_timestamp.apply(lambda x: string2ts(x))


    # USERS
    df_users = pd.read_csv(f"{DATA_DIR}/users.csv")

    # TODO: just temporary, later remove
    #additional_rows = ["user_zipcode"]
    #df_users = df_users.drop(additional_rows, axis=1)

    num2occupation = dict(enumerate(df_users.user_occupation.unique()))
    occupation2num = {y:x for x,y in num2occupation.items()}
    num2gender = dict(enumerate(df_users.user_gender.unique()))
    gender2num = {y:x for x,y in num2gender.items()}
    df_users.user_occupation = df_users.user_occupation.apply(lambda x: occupation2num[x])
    df_users.user_gender = df_users.user_gender.apply(lambda x: gender2num[x])

    df_posters = pd.read_csv(f"{DATA_DIR}/movie_poster.csv", names=["movie_id_ml", "poster_url"])

    # ALL
    df = pd.merge(df_movies, df_ratings, on="movie_id_ml")
    df = pd.merge(df, df_users, on="user_id")
    df = pd.merge(df, df_posters, on="movie_id_ml")

    # Creating UID, IID, FID
    # movies
    id2movie = dict(enumerate(df.movie_id_ml.unique()))
    movie2id = {y:x for x,y in id2movie.items()}

    # users
    id2user = dict(enumerate(df.user_id.unique()))
    user2id = {y:x for x,y in id2user.items()}

    user_ids = list(df_users.user_id.unique())
    total_users = len(user_ids)
    lengths_sum = 0
    lengths = []

    for i in range(total_users):
        length = random.randint(2, 8)
        
        if lengths_sum+length > total_users:
            length = total_users - lengths_sum
            lengths_sum += length
            lengths.append(length)
            break
        elif lengths_sum+length == total_users:
            lengths_sum += length
            lengths.append(length)
            break
        else:
            lengths_sum += length
            lengths.append(length)
            
    friend_ids = [i for i in enumerate(slice_by_lengths(lengths, user_ids))]
    print(f"Number of friend groupd: {len(friend_ids)}, max {max(friend_ids)[0]}")

    user2friendsid = {}
    for fid_and_uids in friend_ids:
        for uid in fid_and_uids[1]:
            user2friendsid[uid] = fid_and_uids[0]

    df["iid"] = df.apply(lambda x: movie2id[x.movie_id_ml], axis=1)
    df["uid"] = df.apply(lambda x: user2id[x.user_id], axis=1)
    df["fid"] = df.apply(lambda x: user2friendsid[x.user_id], axis=1)


    fid2avgage = dict(df.groupby("fid")["user_age"].agg(np.mean))
    fid2medianrating = dict(df.groupby(["fid","iid"])["rating"].agg(np.median))

    df["fid_user_avg_age"] = df.apply(lambda x: fid2avgage[x.fid], axis=1)
    df["rating"] = df.apply(lambda x: fid2medianrating[(x.fid, x.iid)], axis=1)


    df = df.drop(["uid", "user_gender", "user_occupation", "user_age", "user_id", "rating_timestamp"], axis=1)

    df = df.drop_duplicates()


    # shape [n_users, n_user_features]
    df_friends = df[['fid', 'fid_user_avg_age']].drop_duplicates()
    print(f"Number of friends features: {df_friends.shape[0]}")

    df_movies = df[['iid', 'unknown', 'action', 'adventure', 'animation', 'childrens', 'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'noir', 'horror', 'musical', 'mystery', 'romance', 'scifi', 'thriller', 'war', 'western']].drop_duplicates()
    print(f"Number of movies features: {df_movies.shape[0]}")

    return df, df_friends, df_movies, len(friend_ids)

def update_data(friends_id, ratings, rated_movie_ids, df, df_friends, df_movies):
    df_friends = df_friends.append({"fid": friends_id, "fid_user_avg_age":0}, ignore_index=True)
    print(f"New number of friends features: {df_friends.shape[0]}")
    print(f"New number of movies features: {df_movies.shape[0]}")

    data_new_friends_training = []
    for mid, movie_real_id in enumerate(rated_movie_ids):
        avg_mv_rating = np.median(np.array([user_ratings[mid] for user_ratings in ratings]))
        data_new_friends_training.append([friends_id, movie_real_id, avg_mv_rating]) 

    columns = ["fid", "iid", "rating"]
    # user initial input that will be given to him to rate it before recommendation
    df_new_friends_train = pd.DataFrame(data_new_friends_training, columns=columns)

    df_train = df.copy()
    df_train = pd.concat([df_train, df_new_friends_train], sort=False)

    df_train = df_train[["fid", "iid", "rating"]].astype(np.int64)
    #df_new_friends_train = df_new_friends_train[["fid", "iid", "rating"]].astype(np.int64)

    return df_train, df_friends, df_movies

def onehotencoding2genre(x):
        genres= ['unknown','action','adventure','animation','childrens','comedy','crime','documentary','drama','fantasy','noir','horror','musical','mystery','romance','scifi','thriller','war','western']
        ret_val = []
        for c in genres:
            g = getattr(x, c)
            if g == 1:
                ret_val.append(c)
        return ret_val

def get_trending_movie_ids(k, df):
    df_movie_count_mean = df.groupby(["movie_id_ml", "title"], as_index=False)["rating"].agg(["count", "mean"]).reset_index()
    C = df_movie_count_mean["mean"].mean()
    m = df_movie_count_mean["count"].quantile(0.9)

    def weighted_rating(x, m=m, C=C):
        """Calculation based on the IMDB formula"""
        v = x['count']
        R = x['mean']
        return (v/(v+m) * R) + (m/(m+v) * C)


    
    
    
    df_movies = pd.read_csv(f"{DATA_DIR}/movies_cast_company.csv", encoding='utf8')
    df_movies["cast"] = df_movies["cast"].apply(lambda x: json.loads(x))
    df_movies["company"] = df_movies["company"].apply(lambda x: json.loads(x))
    df_movies["genres"] = df_movies.apply(lambda x: onehotencoding2genre(x), axis=1)


    df_movies_1 = df_movie_count_mean.copy().loc[df_movie_count_mean["count"] > m]
    df = pd.merge(df_movies, df_movies_1, on=["movie_id_ml", "title"])


    # Define a new feature 'score' and calculate its value with `weighted_rating()`
    df['score'] = df.apply(weighted_rating, axis=1)


    #Sort movies based on score calculated above
    df = df.sort_values('score', ascending=False).reset_index()

    df = df.head(50)

    df = df.sample(k)

    return list(df.movie_id_ml)