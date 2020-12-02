from __future__ import print_function

import math
from pyspark.sql import SparkSession
from pyspark.sql import functions as f

class ItemCF:
    def __init__(self, spark_session, movies_path, ratings_path):
        self.spark = spark_session
        self.sc = spark_session.sparkContext
        self.movies_df = self.spark.read.option("header", True).csv(movies_path)
        self.ratings_df = self.spark.read.option("header", True).csv(ratings_path)
        self.ratings_df = self.ratings_df.withColumn("userId", self.ratings_df["userId"].cast("int")).withColumn("movieId", self.ratings_df["movieId"].cast("int")).withColumn("rating", self.ratings_df["rating"].cast("float"))
        self.movies_df.withColumn("movieId", self.movies_df["movieId"].cast("int"))
    
    def movieSimilarity(self, id1, id2):
        # calculate the cosine similarity between movies
        # if rating is larger than 3, it means like
        # like id1
        ratings1 = self.ratings_df.filter((self.ratings_df["movieId"] == id1) & (self.ratings_df["rating"] >= 3.0))
        counter1 = ratings1.count()
        # like id2
        ratings2 = self.ratings_df.filter((self.ratings_df["movieId"] == id2) & (self.ratings_df["rating"] >= 3.0))
        counter2 = ratings2.count()
        # like both
        merged_ratings = ratings1.select("userId").intersect(ratings2.select("userId"))
        counter3 = merged_ratings.count()
        if counter2 == 0 or counter1 == 0:
            return 0.0
        return (counter3) / math.sqrt(counter1 * counter2)