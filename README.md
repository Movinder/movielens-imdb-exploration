# Movinder

<div align="center">
  <p>
  <img src="images/movie.gif" width="350" />
  </p>
  <p>
    <a href="">
      <img alt="First release" src="https://img.shields.io/badge/release-v1.0-brightgreen.svg" />
    </a>
  </p>

  <p>
    <strong>Movinder</strong>
  </p>
  
  <p>
    <a href="https://movinder.herokuapp.com/">
      Website
    </a>
  </p>
</div>

This repository contains data exploration of MovieLens and IMDb datasets to create Movinder - movie recommendation system for a group of people.

## Proposal
Using the wealth of data available on user preferences, researchers and online streaming
companies have extensively researched and deployed movie recommender systems.
Most recommendation algorithms process the preferences of each user, and potentially
those of other similar users, to predict other movies they might like. However, watching
movies is often a social activity shared by multiple actors, such as a group of friends. The
social dimension of watching movies contradicts common approaches that strive to
satisfy one user at a time. In this project, we propose a movie recommender system that
aims to maximize the collective satisfaction of a group of users.

In order to implement our movie recommender system, we use the [MovieLens dataset](https://grouplens.org/datasets/movielens/).
The data contains 100K ratings from 1K users on 1.7K movies and has been used
traditionally for recommender system research. Based on the data, we construct two
graphs as follows: i) a user graph that models the similarity of users based on their
personal details (age, gender, occupation and location), and ii) a movie graph that models
the similarity of movies based on their characteristics (release date, genre and title).
Nodes have a vector signal that corresponds to the respective ratings. We plan to enrich
the movie graph with features extracted from the [IMDb dataset](https://datasets.imdbws.com/) (Internet Movie
Database) which contains additional information such as the cast of the movie,
production companies and relationships between movies. The graph structure is used to
generalize the partial information provided by the ratings.

As mentioned, the main goal of our project is to build a recommendation system using
graph neural networks. Moreover, the dataset will enable us to have some insights about
movie tastes in different countries and among different age groups. Using various
dimensionality reduction and clustering algorithms, we may reveal any polarization in the
graph.

In order to recommend the correct movies to the group of users, we need to define a similarity
metric between the two movies (graph where movie is a node) or the two groups of users (graph
where friends are a node). Therefore, since it is not known to be a familiar distance (e.g.
Euclidean), we would need to learn this distance metric which will represent the edges of
the sparsely connected similarity graph. To do that, we would need to create a dataset of
ground truths containing similar movies/users from which the neural network will learn
the similarity metric. Ideally, with this we would be able to learn not only the similarity
metric, but also the embedding space where the movies/users will be visually close to
other similar movies/users.

In addition, by determining the type of graphs, we expect to determine which graph has a
better results when used as a recommender tool. If we have enough time, we are aiming
to build a basic web app for movie recommendations for groups of people.

## Movinder is Online !
You can access the initial version of Movinder by clicking the following link: [Movinder](https://movinder.herokuapp.com)

## Data Exploration Reports
Bellow we have a profiler report of the 5 dataframes. They represent the result of merging the Movielens and IMDb datasets:
- [Cast](https://htmlpreview.github.io/?https://github.com/Movinder/movielens-imdb-exploration/blob/master/data/reports/cast_report.html)
- [Companies](https://htmlpreview.github.io/?https://github.com/Movinder/movielens-imdb-exploration/blob/master/data/reports/companies_report.html)
- [Movies](https://htmlpreview.github.io/?https://github.com/Movinder/movielens-imdb-exploration/blob/master/data/reports/movies_report.html)
- [Ratings](https://htmlpreview.github.io/?https://github.com/Movinder/movielens-imdb-exploration/blob/master/data/reports/ratings_report.html)
- [Users](https://htmlpreview.github.io/?https://github.com/Movinder/movielens-imdb-exploration/blob/master/data/reports/users_report.html)

## Notebooks
Bellow you can find notebooks that were used to do data exploration as well as create different neural networks:
- [Data Analysis](https://nbviewer.jupyter.org/github/Movinder/movielens-imdb-exploration/blob/master/data_analysis.ipynb) - notebook includes the analysis of two datasets we used: MovieLens and IMDb. In the end we combine them and create the final dataframes that will be used in our recommendation models.
- [Trending Movies Recommendation](https://nbviewer.jupyter.org/github/Movinder/movielens-imdb-exploration/blob/master/trending_movie_recommendation.ipynb) - notebook used to find trending movies using the weighted rating. This will be later used for the website to propose users some trending movies (i.e. relarivelly popular) so there is a bigger chance they watched it and have an opinion to give any rating.
- [Doc2Vector]
- [Movie Recommendation using the Colaborative Filtering](https://nbviewer.jupyter.org/github/Movinder/movielens-imdb-exploration/blob/master/collaborative_filtering_can.ipynb) - notebook that implements collaborative filtering methods to recommend movies
- [Movie Recommendation using the Matrix Factorization](https://nbviewer.jupyter.org/github/Movinder/movielens-imdb-exploration/blob/master/matrix_factorization.ipynb) - notebook that implements the matrix factorization for movie recommendation that is also used for the website 
- [Movie Recommendation using the Siamese Neural Network vs. LightFM network for Single Users](https://nbviewer.jupyter.org/github/Movinder/movielens-imdb-exploration/blob/master/movie_recommendation_with_LightFM_person.ipynb) - testing difference in performance between implemented Siamese Neural Network and LighFM network on the data that is dealing only with single users
- [Movie Recommendation using the LightFM network for Groups of Friends](https://nbviewer.jupyter.org/github/Movinder/movielens-imdb-exploration/blob/master/movie_recommendation_with_LightFM_friends.ipynb) - implementing the LightFM on the data that is dealing with groups of users
- [Movie Recommendation for Website](https://nbviewer.jupyter.org/github/Movinder/movielens-imdb-exploration/blob/master/movie_recommendation_with_LightFM_friends_WEBAPP.ipynb) - network implementation that will be used on the website as a Siamese Neural Network option.



## Information
**Course:** [EE-558: A Network Tour of Data Science](https://github.com/mdeff/ntds_2019)
**Team:** Team 6  
**Members:**  Can Yilmaz Altinigne, Jelena Banjac, Sofia Kypraiou, Panagiotis Sioulas 
