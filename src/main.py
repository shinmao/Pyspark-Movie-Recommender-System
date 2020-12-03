from __future__ import print_function
from als_recommender import ALSRecommder
from itemCF import ItemCF

from pyspark import SparkContext
from pyspark.sql import SparkSession
import numpy as np
import os, csv, sys, time
import pandas as pd
from scipy.sparse import csr_matrix
from pyspark.sql.functions import first
from pyspark.sql import functions as F
from pyspark.sql import Row, SQLContext
from pyspark.ml.evaluation import RegressionEvaluator

if __name__ == '__main__':
    # path
    root_path = "/user/hchen28/input"
    ratings_path = root_path + "/ml-20m/ratings.csv"
    movies_path = root_path + "/ml-20m/movies.csv"
    tags_path = root_path + "/ml-20m/tags.csv"
    output_root_path = "/home/hchen28/output"
    als_output_path = output_root_path + "/als_movie_recommendation.txt"
    hybrid_output_path = output_root_path + "/hybrid_movie_recommendation.txt"

    spark = SparkSession.builder.master("local").appName("Movie_Recommender").getOrCreate()
    sc = spark.sparkContext

    movies_df = spark.read.option("header", True).csv(movies_path)
    movies_df = movies_df.withColumn("movieId", movies_df["movieId"].cast("int"))
    ratings_df = spark.read.option("header", True).csv(ratings_path)
    ratings_df = ratings_df.withColumn("userId", ratings_df["userId"].cast("int")).withColumn("movieId", ratings_df["movieId"].cast("int")).withColumn("rating", ratings_df["rating"].cast("float"))
    ratings_df = ratings_df.drop("timestamp")

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
    
    # ALS model
    als_recommender = ALSRecommder(spark, movies_path, ratings_path)
    als_recommender.tune_test([0.1], [x for x in range(6, 20, 2)])
    
    als = als_recommender.set_Params(10, 0.01, 16)
    model = als.fit(new_ratings_df)
    userRecs = model.recommendForAllUsers(10)
    # movieId recommended by ALS
    # { movieId: ALS score }
    als_recommend_list = {}
    # recommended movies for me
    for i in userRecs.collect():
        if i['userId'] == 0:
            with open(als_output_path, "a+") as output:
                output.write("With ALS Recommendation: \n")
                for j in i['recommendations']:
                    output.write("Recommend movieId: " + str(j[0]) + " with score: " + str(j[1]) + " to me.\n")
                    als_recommend_list[j[0]] = j[1]
            output.close()
    
    # item-based CF
    itemcf = ItemCF(spark, movies_path, ratings_path)
    ratedId = [9, 23, 66, 145, 153]
    # testId collects the recommended movieId by ALS model
    testId = [key for key in als_recommend_list.keys()]
    for rated in ratedId:
        for tid in testId:
            sim = itemcf.movieSimilarity(rated, tid)
            als_recommend_list[tid] = als_recommend_list[tid] * sim

    with open(hybrid_output_path, "a+") as output:
        output.write("With hybrid Recommendation: \n")
        sorted_orders = sorted(als_recommend_list.items(), key = lambda x: x[1], reverse=True)
        for i in sorted_orders:
            print(i[0], i[1])
            output.write("Recommend movieId: " + str(i[0]) + "with score: " + str(i[1]) + " to me.\n")
    output.close()

    spark.stop()