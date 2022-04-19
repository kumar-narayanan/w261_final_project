# Databricks notebook source
from pyspark.sql.functions import col,split,isnan, when, count, concat, lit, desc, floor, max, avg, min, regexp_replace, first
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import random
from pyspark.sql.types import IntegerType, array
from pyspark.sql import SQLContext
from pyspark.sql import DataFrameNaFunctions
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Binarizer
from pyspark.ml.feature import OneHotEncoder, VectorAssembler, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

from pyspark.sql.functions import col, max

blob_container = "team08-container" # The name of your container created in https://portal.azure.com
storage_account = "team08" # The name of your Storage account created in https://portal.azure.com
secret_scope = "team08-scope" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "team08-key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# COMMAND ----------

df_weather_airline_3m = spark.read.parquet(f"{blob_url}/df_airlines_weather_3m")
df_weather_airline_6m = spark.read.parquet(f"{blob_url}/df_airlines_weather_6m")

# COMMAND ----------

def simplify_dataframe(dataframe):

    #missing WIND_TYPE
    relevant_columns = ["FL_DATE_UTC", "DEP_HOUR_UTC", "OP_CARRIER_FL_NUM", "ORIGIN", "DEST", "WIND_DIR", "WIND_TYPE", "WIND_SPEED_RATE", "CEILING_HEIGHT", "VIS_DISTANCE", "AIR_TEMP", "DEW_POINT_TEMPERATURE", "SEA_LEVEL_PRESSURE", "DEP_DEL15"]

    converted_df = dataframe.withColumn("WIND_DIR", dataframe["WIND_DIR"].cast(IntegerType())).withColumn("WIND_SPEED_RATE", dataframe["WIND_SPEED_RATE"].cast(IntegerType())).withColumn("CEILING_HEIGHT", dataframe["CEILING_HEIGHT"].cast(IntegerType())).withColumn("VIS_DISTANCE", dataframe["VIS_DISTANCE"].cast(IntegerType())).withColumn("AIR_TEMP", dataframe["AIR_TEMP"].cast(IntegerType())).withColumn("DEW_POINT_TEMPERATURE", dataframe["DEW_POINT_TEMPERATURE"].cast(IntegerType())).withColumn("SEA_LEVEL_PRESSURE", dataframe["SEA_LEVEL_PRESSURE"].cast(IntegerType()))
    
    cleaned_df = converted_df.na.replace(999, 0, subset=["WIND_DIR"])\
                             .na.fill(0, subset=["WIND_DIR"])\
                             .na.replace(99999, 0, subset=["CEILING_HEIGHT"])\
                             .na.replace(999999, 0, subset=["VIS_DISTANCE"])\
                             .na.fill(0, subset=["VIS_DISTANCE"])\
                             .na.replace(9999, 0, subset=["AIR_TEMP"])\
                             .na.replace(9999, 0, subset=["DEW_POINT_TEMPERATURE"]) \
                             .na.replace(9999, 0, subset=["SEA_LEVEL_PRESSURE"])
    
    simple_dataframe = cleaned_df.select([col for col in relevant_columns])
    
    simple_dataframe = simple_dataframe.groupBy(simple_dataframe.FL_DATE_UTC, simple_dataframe.DEP_HOUR_UTC, simple_dataframe.OP_CARRIER_FL_NUM, simple_dataframe.ORIGIN) \
                               .agg(max("DEST").alias("DEST"), \
                                    max("WIND_DIR").alias("WIND_DIR"), \
                                    max("WIND_SPEED_RATE").alias("WIND_SPEED_RATE"), \
                                    max("CEILING_HEIGHT").alias("CEILING_HEIGHT"), \
                                    max("VIS_DISTANCE").alias("VIS_DISTANCE"), \
                                    max("AIR_TEMP").alias("AIR_TEMP"), \
                                    max("DEW_POINT_TEMPERATURE").alias("DEW_POINT_TEMPERATURE"), \
                                    max("SEA_LEVEL_PRESSURE").alias("SEA_LEVEL_PRESSURE"), \
                                    max("DEP_DEL15").alias("DEP_DEL15"))
                                 
    return simple_dataframe

# COMMAND ----------

def prepare_dataframe(dataframe):
#     binarizer = Binarizer(threshold=0.1, inputCol="DEP_DEL15", outputCol="Delay")
#     binarized_df = binarizer.transform(dataframe)
    feature_columns = ["WIND_DIR", "WIND_SPEED_RATE", "CEILING_HEIGHT", "VIS_DISTANCE", "AIR_TEMP", "DEW_POINT_TEMPERATURE", "SEA_LEVEL_PRESSURE"]
    assembler = VectorAssembler(inputCols = feature_columns, outputCol = "features")
    assembled_df = assembler.transform(dataframe)
    return assembled_df

# COMMAND ----------

def train_test_split(dataframe, train_weight, test_weight, seed = 2018):
    dates = dataframe.select(dataframe.FL_DATE_UTC).distinct().orderBy(dataframe.FL_DATE_UTC)
    
    train_dates, test_dates = dates.randomSplit([train_weight, test_weight], seed)
    train_dates = set(train_dates.toPandas()['FL_DATE_UTC'])
    test_dates = set(test_dates.toPandas()['FL_DATE_UTC'])
    
    train_df = dataframe.where(dataframe.FL_DATE_UTC.isin(train_dates))
    test_df = dataframe.where(dataframe.FL_DATE_UTC.isin(test_dates))
    
    return train_df, test_df

# COMMAND ----------

def calculate_f1_score(dataframe):
    TP = dataframe.where((dataframe.DEP_DEL15 == 1) & (dataframe.prediction == 1)).count()
    FP = dataframe.where((dataframe.DEP_DEL15 == 0) & (dataframe.prediction == 1)).count()
    TN = dataframe.where((dataframe.DEP_DEL15 == 0) & (dataframe.prediction == 0)).count()
    FN = dataframe.where((dataframe.DEP_DEL15 == 1) & (dataframe.prediction == 0)).count()
    f1 = np.multiply(2, np.divide(TP, (TP + 0.5*(FP+FN)) ))
    return f1

# COMMAND ----------

def N_fold_Validation(N, dataframe, train_weight, test_weight):
    scores = []
    random.seed(2018)
    seeds = random.sample(range(1, 3000), N)
    
    for i in range(N):
        seed = seeds[i]
        train_df, test_df = train_test_split(dataframe, train_weight, test_weight, seed)
        dt = DecisionTreeClassifier(labelCol="DEP_DEL15", featuresCol="features", maxDepth=25, minInstancesPerNode=30, impurity="gini")
        pipeline = Pipeline(stages=[dt])
        model = pipeline.fit(train_df)
        predictions = model.transform(test_df)
        pred_labels = predictions.select(predictions.DEP_DEL15, predictions.prediction)
        f1 = calculate_f1_score(pred_labels)
        print("Fold " + str(i+1) + ": " + str(f1))
        scores.append(f1)
    return np.mean(scores)
        

# COMMAND ----------

simplified_dataframe_3m = simplify_dataframe(df_weather_airline_3m)
prepared_dataframe_3m = prepare_dataframe(simplified_dataframe_3m)
f1_score_3m = N_fold_Validation(5, prepared_dataframe_3m, 0.8, 0.2)
print("3-month F1 Score: " + str(f1_score_3m))

# COMMAND ----------

simplified_dataframe_6m = simplify_dataframe(df_weather_airline_6m)
prepared_dataframe_6m = prepare_dataframe(simplified_dataframe_6m)
f1_score_6m = N_fold_Validation(5, prepared_dataframe_6m, 0.8, 0.2)
print("6-month F1 Score: " + str(f1_score_6m))

# COMMAND ----------

# MAGIC %md
# MAGIC #FULL DATAFRAME BELOW THIS

# COMMAND ----------

df_airlines_weather_full = spark.read.parquet(f"{blob_url}/df_airlines_weather")
df_airlines_weather = df_airlines_weather_full.where(df_airlines_weather_full.FL_DATE_UTC < '2019-01-01')
print(df_airlines_weather_full.count())
print(df_airlines_weather.count())

# COMMAND ----------

display(df_airlines_weather)

# COMMAND ----------

def simplify_dataframe_full(df):
    df = df.where((df.DEP_DEL15).isNotNull())
    
    relevant_columns = ["FL_DATE_UTC", "DEP_HOUR_UTC", "OP_CARRIER_FL_NUM", "ORIGIN", "DEST", "ELEVATION", "WIND_GUST_SPEED_RATE", "WIND_DIR_IMP", "WIND_SPEED_RATE_IMP", "VIS_DISTANCE_IMP", "CEILING_HEIGHT_IMP", "AIR_TEMP_IMP", "DEW_POINT_TEMPERATURE_IMP", "SEA_LEVEL_PRESSURE_IMP", "ATMOS_PRESSURE_3H_IMP", "PRECIPITATION_RATE", "PRESENT_ACC_OHE", "PAST_ACC_OHE", "ATMOS_PRESSURE_TENDENCY_3H_OHE", "WIND_TYPE_OHE", "DEP_DEL15"]

    simplified_dataframe = df.select([col for col in relevant_columns])
    simplified_dataframe = simplified_dataframe.groupBy(simplified_dataframe.FL_DATE_UTC, simplified_dataframe.DEP_HOUR_UTC, simplified_dataframe.OP_CARRIER_FL_NUM, simplified_dataframe.ORIGIN) \
                                   .agg(max("DEST").alias("DEST"), \
                                        avg("ELEVATION").alias("ELEVATION"), \
                                        avg("WIND_GUST_SPEED_RATE").alias("WIND_GUST_SPEED_RATE"),\
                                        avg("WIND_DIR_IMP").alias("WIND_DIR_IMP"),\
                                        avg("WIND_SPEED_RATE_IMP").alias("WIND_SPEED_RATE_IMP"),\
                                        avg("VIS_DISTANCE_IMP").alias("VIS_DISTANCE_IMP"),\
                                        avg("CEILING_HEIGHT_IMP").alias("CEILING_HEIGHT_IMP"),\
                                        avg("AIR_TEMP_IMP").alias("AIR_TEMP_IMP"),\
                                        avg("DEW_POINT_TEMPERATURE_IMP").alias("DEW_POINT_TEMPERATURE_IMP"),\
                                        avg("SEA_LEVEL_PRESSURE_IMP").alias("SEA_LEVEL_PRESSURE_IMP"),\
                                        avg("ATMOS_PRESSURE_3H_IMP").alias("ATMOS_PRESSURE_3H_IMP"),\
                                        avg("PRECIPITATION_RATE").alias("PRECIPITATION_RATE"),\
                                        first("PRESENT_ACC_OHE").alias("PRESENT_ACC_OHE"),\
                                        first("PAST_ACC_OHE").alias("PAST_ACC_OHE"),\
                                        first("ATMOS_PRESSURE_TENDENCY_3H_OHE").alias("ATMOS_PRESSURE_TENDENCY_3H_OHE"),\
                                        first("WIND_TYPE_OHE").alias("WIND_TYPE_OHE"),\
                                        max("DEP_DEL15").alias("DEP_DEL15"))
    return simplified_dataframe

# COMMAND ----------

def prepare_dataframe_full(df):
    non_null_df = df.where((df.ELEVATION).isNotNull())
    feature_columns = ["ELEVATION", "WIND_GUST_SPEED_RATE", "WIND_DIR_IMP", "WIND_SPEED_RATE_IMP", "VIS_DISTANCE_IMP", "CEILING_HEIGHT_IMP", "AIR_TEMP_IMP", "DEW_POINT_TEMPERATURE_IMP", "SEA_LEVEL_PRESSURE_IMP", "ATMOS_PRESSURE_3H_IMP", "PRECIPITATION_RATE", "PRESENT_ACC_OHE", "PAST_ACC_OHE", "ATMOS_PRESSURE_TENDENCY_3H_OHE", "WIND_TYPE_OHE"]
    assembler = VectorAssembler(inputCols = feature_columns, outputCol = "features")
    non_null_df = assembler.transform(non_null_df)
    return non_null_df

# COMMAND ----------

s_dataframe = simplify_dataframe_full(df_airlines_weather)
# display(s_dataframe)
p_dataframe = prepare_dataframe_full(s_dataframe)
# display(p_dataframe)
f1_score = N_fold_Validation(5, p_dataframe, 0.8, 0.2)
print("F1 Score: " + str(f1_score))

# COMMAND ----------

display(s_dataframe.where((s_dataframe.OP_CARRIER_FL_NUM == 668) & (s_dataframe.FL_DATETIME_UTC == "2015-01-01T06:55:00.000+0000")))
display(p_dataframe.where((p_dataframe.OP_CARRIER_FL_NUM == 668) & (p_dataframe.FL_DATETIME_UTC == "2015-01-01T06:55:00.000+0000")))

# COMMAND ----------

test = s_dataframe.where((s_dataframe.ELEVATION).isNotNull())
display(test)

# COMMAND ----------

test_2 = s_dataframe.where((s_dataframe.ELEVATION).isNull()).withColumn("features", lit(None))
display(test_2)

# COMMAND ----------

print(s_dataframe.where((s_dataframe.DEP_DEL15).isNull()).count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### TODO
# MAGIC **Needed for Thursday**
# MAGIC 
# MAGIC 1. How to handle records with no weather? Ignore? Use latest weather reading? (will require rejoin)
# MAGIC    11. Ignore for now. 0.1% of data. 
# MAGIC 1. **SMOTE** (Kumar)
# MAGIC 1. **Logistic Regression** (Kumar)
# MAGIC 1. **Hyperopt** (Mrinal)
# MAGIC 1. **Gap Analysis** (Sushant)
# MAGIC 1. **Slides for Thursday** (Ajeya)
# MAGIC 1. **PCA** (Mrinal)
# MAGIC 1. Random Forest (Ajeya)
# MAGIC 1. NaiveBayes (see what happens) (Ajeya)
# MAGIC 1. Feature Importance (Sushant)
# MAGIC 1. Gradient Boosted Trees (Mrinal)

# COMMAND ----------

# MAGIC %md
# MAGIC #HYPERPARAMETER TUNING

# COMMAND ----------

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.early_stop import no_progress_loss

# COMMAND ----------

def N_fold_Validation_tuning(N, dataframe, train_weight, test_weight, max_depth, min_inst_per_node, impurity):
    scores = []
    random.seed(2018)
    seeds = random.sample(range(1, 3000), N)
    
    for i in range(N):
        seed = seeds[i]
        train_df, test_df = train_test_split(dataframe, train_weight, test_weight, seed)
        dt = DecisionTreeClassifier(labelCol="DEP_DEL15", featuresCol="features", maxDepth=max_depth, minInstancesPerNode=min_inst_per_node, impurity=impurity)
        pipeline = Pipeline(stages=[dt])
        model = pipeline.fit(train_df)
        predictions = model.transform(test_df)
        pred_labels = predictions.select(predictions.DEP_DEL15, predictions.prediction)
        f1 = calculate_f1_score(pred_labels)
        print("Fold " + str(i+1) + ": " + str(f1))
        scores.append(f1)
    return np.mean(scores)

# COMMAND ----------

s_dataframe = simplify_dataframe_full(df_airlines_weather)
p_dataframe = prepare_dataframe_full(s_dataframe)
p_dataframe = p_dataframe.where(p_dataframe.FL_DATE_UTC < '2016-01-01')

# COMMAND ----------

parameter_ranges = {'max_depth' : range(5,30), 'min_inst_per_node' : range(10,1000,10), 'impurity': ['entropy', 'gini']}
parameter_space = {'max_depth' : hp.choice('max_depth', parameter_ranges['max_depth']), 'min_inst_per_node': hp.choice('min_inst_per_node', parameter_ranges['min_inst_per_node']), 'impurity': hp.choice('impurity', parameter_ranges['impurity'])}

# COMMAND ----------

# f1_score = N_fold_Validation_tuning(1, p_dataframe, 0.8, 0.2, 25, 30)
trials = Trials()
best = fmin(lambda x: 1 - N_fold_Validation_tuning(1, p_dataframe, 0.8, 0.2, x['max_depth'], x['min_inst_per_node'], x['impurity']), parameter_space, algo=tpe.suggest, max_evals=100, trials=trials)

# COMMAND ----------

for key,value in best.items():
    print(key + ":" + str(parameter_ranges[key][value]))

# COMMAND ----------

print(trials.vals)
print(trials.results)

# COMMAND ----------

print(parameter_space)

# COMMAND ----------

