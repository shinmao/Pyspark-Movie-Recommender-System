## Pyspark Movie Recommender System
The project implements a hybrid system which uses ALS solution and item-item CF to introduce movie to users. What's the most important, the recommendation algorithm is not based on genres of movies themselves, but based on the ratings given by users. In this project, after training model based on the Movie Lens 20M dataset, I would also give my ratings to several movies, which is my utility matrix, and add it to the dataset. At the end, the hybrid system would give me a list of recommended movies.

## What's ALS? What's item-item CF?
Here are the documentation I used to build up this system:  
* [Spark 3.0.1 Collaborative Filtering](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html)