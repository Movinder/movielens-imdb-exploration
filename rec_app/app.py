from flask import Flask, request, render_template, session
import numpy as np
import pandas as pd
from surprise import NMF, Dataset, Reader
from scipy.stats import hmean 
import os

app = Flask(__name__, template_folder='templates')
app.secret_key = "super secret key"
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


def recommendation(userArray, numUsers, movieIds):

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

    return list(zip(list(df_ML_movies[df_ML_movies.movie_id_ml.isin(movie_ind)].title), list(scores)))


@app.route('/', methods=['GET', 'POST'])
def main():

    if request.method == 'POST':
        
        if 'run-model' in request.form:
            pu = recommendation(session['arr'], session['members'], session['movieIds'])
            session.clear()
            session['counter'] = -1
            session['members'] = 0
            session['movieIds'] = [int(x) for x in (np.random.choice(movies_[-200:], 15, replace=False))]
            session['top15'] = list(df_ML_movies[df_ML_movies.movie_id_ml.isin(session['movieIds'])].title)
            session['arr'] = None
            return(render_template('main.html', settings = {'showVote': False, 'people': 0, 'buttonDisable': False, 'recommendation': pu}))
        else:
            if 'people-select' in request.form:
                count = int(request.form.get('people-select'))
                session['members'] = count
                session['arr'] = [[0 for x in range(15)] for y in range(count)]
                return(render_template('main.html', settings = {'showVote': True, 'people': count, 'buttonDisable': True, 'recommendation': None}))

            elif 'person-select-0' in request.form:
                for i in range(session['members']):
                    session['arr'][i][session['counter'] + 1] = int(request.form.get('person-select-{}'.format(i)))
                
                session['counter'] += 1      
                return(render_template('main.html', settings = {'showVote': True, 'people': len(request.form), 'buttonDisable': True, 'recommendation': None}))

    elif request.method == 'GET':
        session.clear()
        session['counter'] = -1
        session['members'] = 0
        session['movieIds'] = [int(x) for x in (np.random.choice(movies_[-200:], 15, replace=False))]
        session['top15'] = list(df_ML_movies[df_ML_movies.movie_id_ml.isin(session['movieIds'])].title)
        session['arr'] = None

        return(render_template('main.html', settings = {'showVote': False, 'people': 0, 'buttonDisable': False, 'recommendation': None}))
    
if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)