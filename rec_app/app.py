from flask import Flask, request, render_template, session
import numpy as np
import pandas as pd
from surprise import NMF, Dataset, Reader
from scipy.stats import hmean 
import os

from src.data import initial_data, get_trending_movie_ids, update_data, onehotencoding2genre
from src.siamese_training import training

app = Flask(__name__, template_folder='templates')
app.secret_key = "super secret key"

DATA_DIR = "../../movielens-imdb-exploration/data"

df, df_friends, df_movies, new_fid = initial_data()
df["genres"] = df.apply(lambda x: onehotencoding2genre(x), axis=1)
print(df.columns)
df_movie_urls = df[["iid", "movie_id_ml", "poster_url", "title"]].drop_duplicates()
trending_movie_ids = get_trending_movie_ids(15, df)

ratings = pd.read_csv('static/ratings.csv')
mat = np.zeros((max(ratings.user_id), max(ratings.movie_id_ml)))
ind = np.array(list(zip(list(ratings.user_id-1), list(ratings.movie_id_ml-1))))
mat[ind[:,0], ind[:,1]] = 1
movies_ = mat.sum(axis=0).argsort()+1
column_item = ["movie_id_ml", "title", "release", "vrelease", "url", "unknown", 
					"action", "adventure", "animation", "childrens", "comedy",
				   "crime", "documentary", "drama", "fantasy", "noir", "horror",
				   "musical", "mystery", "romance", "scifi", "thriller",
				   "war", "western"]
df_ML_movies = pd.read_csv('static/u.item.txt', delimiter='|', encoding = "ISO-8859-1", names=column_item) 
df_posters = pd.read_csv(f"{DATA_DIR}/movie_poster.csv", names=["movie_id_ml", "poster_url"])
df_ML_movies = pd.merge(df_ML_movies,df_posters, on="movie_id_ml")




def recommendation_mf(userArray, numUsers, movieIds):

	ratings_dict = {'itemID': list(ratings.movie_id_ml) + list(numUsers*movieIds),
					'userID': list(ratings.user_id) + [max(ratings.user_id)+1+x for x in range(numUsers) for y in range(15)],
					'rating': list(ratings.rating) + [item for sublist in userArray for item in sublist]
				}

	df = pd.DataFrame(ratings_dict)
	reader = Reader(rating_scale=(1, 5))
	data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
	trainset = data.build_full_trainset()

	nmf = NMF()
	nmf.fit(trainset)

	userIds = [trainset.to_inner_uid(max(ratings.user_id)+1+x) for x in range(numUsers)]

	mat = np.dot(nmf.pu, nmf.qi.T)

	scores = hmean(mat[userIds, :], axis=0)
	best_movies = scores.argsort()
	best_movies = best_movies[-9:][::-1]
	scores = scores[best_movies]
	movie_ind = [trainset.to_raw_iid(x) for x in best_movies]

	recommendation = list(zip(list(df_ML_movies[df_ML_movies.movie_id_ml.isin(movie_ind)].title), 
					list(df_ML_movies[df_ML_movies.movie_id_ml.isin(movie_ind)].poster_url), 
					list(scores)))
	print(recommendation)
	print(len(recommendation[0]))
	return recommendation

def recommendation_siamese(top_movies, scores):
	recommendation = list(zip(list(top_movies.title), 
					list(top_movies.poster_url), 
					scores)) 
	return recommendation



@app.route('/', methods=['GET', 'POST'])
def main():

	if request.method == 'POST':
		
		# Get recommendations!
		if 'run-mf-model' in request.form:
			pu = recommendation_mf(session['arr'], session['members'], session['movieIds'])
			session.clear()
			trending_movie_ids = get_trending_movie_ids(15, df)
			session['counter'] = -1
			session['members'] = 0
			session['movieIds'] = list(df_movie_urls[df_movie_urls.movie_id_ml.isin(trending_movie_ids)].movie_id_ml)
			session['top15'] = list(df_movie_urls[df_movie_urls.movie_id_ml.isin(trending_movie_ids)].title) 
			session['top15_posters'] = list(df_movie_urls[df_movie_urls.movie_id_ml.isin(trending_movie_ids)].poster_url)
			session['arr'] = None
			return(render_template('main.html', settings = {'showVote': False, 'people': 0, 'buttonDisable': False, 'recommendation': pu}))
		
		if 'run-siamese-model' in request.form:
			# global df
			global df_friends
			global df_movies
			global new_fid
			df_train, df_friends, df_movies = update_data(new_fid, session['arr'], session['movieIds'], df, df_friends, df_movies)
			
			top_movie_ids, scores = training(df_train, df_friends, df_movies, new_fid)
			top_movies = df_movie_urls[df_movie_urls.iid.isin(top_movie_ids)]

			pu = recommendation_siamese(top_movies, scores)

			return(render_template('main.html', settings = {'showVote': False, 'people': 0, 'buttonDisable': False, 'recommendation': pu}))
		
		# Choose number of people in the group
		elif 'people-select' in request.form:
			count = int(request.form.get('people-select'))
			session['members'] = count
			session['arr'] = [[0 for x in range(15)] for y in range(count)] 
			return(render_template('main.html', settings = {'showVote': True, 'people': count, 'buttonDisable': True, 'recommendation': None}))

		# All people voting
		elif 'person-select-0' in request.form:
			for i in range(session['members']):
				session['arr'][i][session['counter'] + 1] = int(request.form.get(f'person-select-{i}'))
			
			session['counter'] += 1      
			return(render_template('main.html', settings = {'showVote': True, 'people': len(request.form), 'buttonDisable': True, 'recommendation': None}))

	elif request.method == 'GET':
		session.clear()
		#global trending_movie_ids
		trending_movie_ids = get_trending_movie_ids(15, df)
		session['counter'] = -1
		session['members'] = 0
		session['movieIds'] = list(df_movie_urls[df_movie_urls.movie_id_ml.isin(trending_movie_ids)].movie_id_ml) 
		session['top15'] = list(df_movie_urls[df_movie_urls.movie_id_ml.isin(trending_movie_ids)].title) 
		session['top15_posters'] = list(df_movie_urls[df_movie_urls.movie_id_ml.isin(trending_movie_ids)].poster_url)
		session['arr'] = None

		return(render_template('main.html', settings = {'showVote': False, 'people': 0, 'buttonDisable': False, 'recommendation': None}))

@app.route('/static/<path:path>')
def serve_dist(path):
	return send_from_directory('static', path)

if __name__ == '__main__':
	# Bind to PORT if defined, otherwise default to 5000.
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port)