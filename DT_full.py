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

blob_container = "team08-container" # The name of your container created in https://portal.azure.com
storage_account = "team08" # The name of your Storage account created in https://portal.azure.com
secret_scope = "team08-scope" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "team08-key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# COMMAND ----------

df_airlines_weather_full = spark.read.parquet(f"{blob_url}/df_airlines_weather")
df_airlines_weather_full.cache()
df_airlines_weather = df_airlines_weather_full.where(df_airlines_weather_full.FL_DATE_UTC < '2019-01-01')
df_airlines_weather.cache()
df_airlines_weather_19 = df_airlines_weather_full.where(df_airlines_weather_full.FL_DATE_UTC >= '2019-01-01')
df_airlines_weather_19.cache()
print(df_airlines_weather_full.count())
print(df_airlines_weather.count())
print(df_airlines_weather_19.count())

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
    feature_columns = ["ELEVATION", "DEP_HOUR_UTC", "WIND_GUST_SPEED_RATE", "WIND_DIR_IMP", "WIND_SPEED_RATE_IMP", "VIS_DISTANCE_IMP", "CEILING_HEIGHT_IMP", "AIR_TEMP_IMP", "DEW_POINT_TEMPERATURE_IMP", "SEA_LEVEL_PRESSURE_IMP", "ATMOS_PRESSURE_3H_IMP", "PRECIPITATION_RATE", "PRESENT_ACC_OHE", "PAST_ACC_OHE", "ATMOS_PRESSURE_TENDENCY_3H_OHE", "WIND_TYPE_OHE"]
    assembler = VectorAssembler(inputCols = feature_columns, outputCol = "features")
    non_null_df = assembler.transform(non_null_df)
    return non_null_df

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
    print(TP, FP, TN, FN)
    f1 = np.divide(TP, (TP + 0.5*(FP+FN)))
    return f1

# COMMAND ----------

def N_fold_Validation(N, dataframe, train_weight, test_weight):
    max_score = -1
    best_model = None
    
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
        if f1 > max_score:
            max_score = f1
            best_model = model
    return max_score, best_model

# COMMAND ----------

def sample_dataframe(dataframe): 
    delayed_df = dataframe.where(dataframe.DEP_DEL15 == 1)
    non_delayed_df = dataframe.where(dataframe.DEP_DEL15 == 0)
    delayed_df_samp = delayed_df.sample(False, 0.90, seed = 2018)
    non_delayed_df_samp = non_delayed_df.sample(False, 0.20, seed = 2018)
    return delayed_df_samp.union(non_delayed_df_samp)

# COMMAND ----------

s_dataframe = simplify_dataframe_full(df_airlines_weather)
s_dataframe.cache()
p_dataframe = prepare_dataframe_full(s_dataframe)
p_dataframe.cache()
samp_dataframe = sample_dataframe(p_dataframe)
samp_dataframe.cache()
print(s_dataframe.count())
print(p_dataframe.count())
print(samp_dataframe.count())

# COMMAND ----------

# f1_score, model = N_fold_Validation(1, samp_dataframe, 0.8, 0.2)
# print("F1 Score: " + str(f1_score))

# COMMAND ----------

train_df, test_df = train_test_split(samp_dataframe, 0.8, 0.2, 2018)
train_df.cache()
test_df.cache()
print(train_df.count())
print(test_df.count())
# display(test_df)

# COMMAND ----------

dt = DecisionTreeClassifier(labelCol="DEP_DEL15", featuresCol="features", maxDepth=29, minInstancesPerNode=50, impurity="gini", seed=2018)
model = dt.fit(train_df)
predictions = model.transform(test_df)
predictions.cache()
print("TEST: " + str(test_df.count()))
print("PRED: " + str(predictions.count()))

# COMMAND ----------

print(calculate_f1_score(predictions))

# COMMAND ----------

s_dataframe_19 = simplify_dataframe_full(df_airlines_weather_19)
s_dataframe_19.cache()
p_dataframe_19 = prepare_dataframe_full(s_dataframe_19)
p_dataframe_19.cache()


# COMMAND ----------

predictions_19 = model.transform(p_dataframe_19)
predictions_19.cache()

# COMMAND ----------

print(calculate_f1_score(predictions_19))

# COMMAND ----------

p_dataframe_19.count()

# COMMAND ----------

def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))


# COMMAND ----------

ranked_features = ExtractFeatureImp(model.featureImportances, p_dataframe, "features")

# COMMAND ----------

display(ranked_features)

# COMMAND ----------

