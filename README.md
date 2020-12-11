## Pyspark Movie Recommender System
The project implements a hybrid system which uses ALS solution and item-item CF to introduce movie to users. What's the most important, the recommendation algorithm is not based on genres of movies themselves, but based on the ratings given by users. In this project, after training model based on the Movie Lens 20M dataset, I would also give my ratings to several movies, which is my utility matrix, and add it to the dataset. At the end, the hybrid system would give me a list of recommended movies.

## What's ALS? What's item-item CF?
Here are the documentation I used to build up this system:  
* [Spark 3.0.1 Collaborative Filtering](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html)

Class of ALS is in `/src/als_recommender.py`.  
Class of item-item CF is in `/src/itemCF.py`.

## Recommend list from ALS
First I add my personal ratings for several movies:
```python
# 9: Sudden Death (1995)
# 23: Assassins (1995)
# 66: Lawnmower Man 2: Beyond Cyberspace (1996)
# 145: Bad Boys (1995)
# 153: Batman Forever (1995) 
my_rated_df = spark.createDataFrame(
    [
        (0, 9, 4),
        (0, 23, 2),
        (0, 66, 2),
        (0, 145, 4),
        (0, 153, 3)
    ],
    ['userId', 'movieId', 'rating']
)
new_ratings_df = ratings_df.union(my_rated_df)
```
ALS recommend some movies to me based on new ratings:
```
With ALS Recommendation: 
Recommend movieId: 5271 with score: 10.924333572387695 to me.
Recommend movieId: 74263 with score: 9.590706825256348 to me.
Recommend movieId: 71017 with score: 9.150797843933105 to me.
Recommend movieId: 2954 with score: 8.941307067871094 to me.
Recommend movieId: 86947 with score: 8.911588668823242 to me.
Recommend movieId: 104583 with score: 8.319062232971191 to me.
Recommend movieId: 50942 with score: 8.305917739868164 to me.
Recommend movieId: 48045 with score: 8.248882293701172 to me.
Recommend movieId: 32088 with score: 8.155112266540527 to me.
Recommend movieId: 27009 with score: 8.083059310913086 to me.
```

## Conclusion
For more results, you can take a look at my report in `Report.docx`.