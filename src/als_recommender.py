from __future__ import print_function

from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics

class ALSRecommder:
    def __init__(self, spark_session, movies_path, ratings_path):
        self.spark = spark_session
        self.sc = spark_session.sparkContext
        self.movies_df = self.spark.read.option("header", True).csv(movies_path)
        self.ratings_df = self.spark.read.option("header", True).csv(ratings_path)
        self.ratings_df = self.ratings_df.withColumn("userId", self.ratings_df["userId"].cast("int")).withColumn("movieId", self.ratings_df["movieId"].cast("int")).withColumn("rating", self.ratings_df["rating"].cast("float"))
        self.model = ALS(userCol='userId', itemCol='movieId', ratingCol='rating', coldStartStrategy='drop')

    # 10, 0.01, 16
    def set_Params(self, maxIter, reParam, rank):
        self.model = self.model.setMaxIter(maxIter).setRegParam(reParam).setRank(rank)
        return self.model

    def tune_test(self, regParam, ranks):
        train, validation, test = self.ratings_df.randomSplit([0.6, 0.2, 0.2])
        min_err = float('inf')
        best_rank = -1
        best_reg = 0
        final_model = None
        for rank in ranks:
            for reg in regParam:
                als = self.model.setMaxIter(10).setRegParam(reg).setRank(rank)
                model = als.fit(train)
                predictions = model.transform(validation)
                evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
                rmse = evaluator.evaluate(predictions)
                if rmse < min_err:
                    min_err = rmse
                    best_rank = rank
                    best_reg = reg
                    final_model = model
        # get best model "final_model"
        print("Best model RMSE on validation set with rank: " + str(rank) + " is: " + str(rmse))
        self.model = final_model
        # test model
        preds = self.model.transform(test)
        rmse_eval = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        mse_eval = RegressionEvaluator(metricName="mse", labelCol="rating", predictionCol="prediction")

        rmse = rmse_eval.evaluate(preds)
        mse = mse_eval.evaluate(preds)

        with open("/home/hchen28/output/tuned_result.txt", "a+") as result:
            result.write("Test set (RMSE/MSE): " + str(rmse) + "/" + str(mse) + "\n")
        result.close()