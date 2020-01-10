from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
import numpy as np

def _build_interaction_matrix(rows, cols, data):

    mat = sp.lil_matrix((rows, cols), dtype=np.int32)

    for fid, iid, rating in data:
        # Let's assume only really good things are positives
        if rating >= 4.0:
            mat[fid, iid] = 1.0

    return mat.tocoo()

def create_sparse_matrix(df):
    """
    Return (train_interactions, test_interactions).
    """

    fids = set(df.fid.unique())
    iids = set(df.iid.unique())

    rows = max(fids) + 1 
    cols = max(iids) + 1

    print("Friends number: ", len(fids), rows)
    print("Movies number: ", len(iids), cols)

    return _build_interaction_matrix(rows, cols, df.values.tolist())

def train(data, user_features=None, item_features=None, use_features = False):
    loss_type = "warp"  # "bpr"

    model = LightFM(learning_rate=0.05, loss=loss_type, max_sampled=100)

    if use_features:
        model.fit_partial(data, epochs=20, user_features=friends_features, item_features=item_features)
        train_precision = precision_at_k(model, data, k=10, user_features=friends_features, item_features=item_features).mean()
        
        train_auc = auc_score(model, data, user_features=friends_features, item_features=item_features).mean()
        
        print(f'Precision: train {train_precision:.2f}')
        print(f'AUC: train {train_auc:.2f}')
    else:
        model.fit_partial(data, epochs=20)
        
        train_precision = precision_at_k(model, data, k=10).mean()
        
        train_auc = auc_score(model, data).mean()
        
        print(f'Precision: train {train_precision:.2f}')
        print(f'AUC: train {train_auc:.2f}')

    return model


def predict_top_k_movies(model, friends_id, k, data, user_features=None, item_features=None, use_features=False):
    n_users, n_movies = data.shape
    if use_features:
        prediction = model.predict(friends_id, np.arange(n_movies), user_features=friends_features, item_features=item_features)
    else:
        prediction = model.predict(friends_id, np.arange(n_movies))
    
    movie_ids = np.arange(data.shape[1])
    # return movie ids
    return movie_ids[np.argsort(-prediction)][:k], prediction[np.argsort(-prediction)][:k]

def training(df_train, df_friends, df_movies, new_fid):
	train_data = create_sparse_matrix(df_train)

	# shape [n_users, n_user_features]
	friends_features = sp.csr_matrix(df_friends.values)
	item_features = sp.csr_matrix(df_movies.values)

	model = train(train_data, user_features=friends_features, item_features=item_features, use_features = False)

	k = 18
	top_movie_ids, scores = predict_top_k_movies(model, new_fid, k, train_data, user_features=friends_features, item_features=item_features, use_features = False)
	return top_movie_ids, scores
