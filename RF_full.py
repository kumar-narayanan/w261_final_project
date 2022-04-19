# Databricks notebook source
from pyspark.sql.functions import col,split,isnan, when, count, concat, lit, desc, floor, max, avg, min, regexp_replace, first
import numpy as np
import pandas as pd
import random
from pyspark.sql.types import IntegerType, array
from pyspark.sql import SQLContext
from pyspark.sql import DataFrameNaFunctions
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import Binarizer
from pyspark.ml.feature import OneHotEncoder, VectorAssembler, StringIndexer, VectorIndexer, VectorSlicer

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
        rf = RandomForestClassifier(labelCol="DEP_DEL15", featuresCol="features", maxDepth=29, minInstancesPerNode=50, impurity="gini")
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

# df_airlines_weather = df_airlines_weather.where(df_airlines_weather.FL_DATE_UTC < '2016-01-01')
s_dataframe = simplify_dataframe_full(df_airlines_weather)
s_dataframe.cache()
p_dataframe = prepare_dataframe_full(s_dataframe)
p_dataframe.cache()
samp_dataframe = sample_dataframe(p_dataframe)
samp_dataframe.cache()

# COMMAND ----------

train_df, test_df = train_test_split(samp_dataframe, 0.8, 0.2, 2018)
train_df.cache()
test_df.cache()

# COMMAND ----------

rf = RandomForestClassifier(labelCol="DEP_DEL15", featuresCol="features", maxDepth=29, minInstancesPerNode=50, impurity="gini")
model = rf.fit(train_df)
predictions = model.transform(test_df)
predictions.cache()

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

print(model.featureImportances.toArray())

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

# MAGIC %md
# MAGIC Validating for Feature Importance

# COMMAND ----------

#baseline with all features
rf = RandomForestClassifier(labelCol="DEP_DEL15", featuresCol="features", maxDepth=29, minInstancesPerNode=50, impurity="gini")
train_df, test_df = train_test_split(samp_dataframe, 0.8, 0.2)
model = rf.fit(train_df)
predictions = model.transform(test_df)
predictions.cache()

# COMMAND ----------

#get list of feature importances
ranked_features = ExtractFeatureImp(model.featureImportances, samp_dataframe, "features")

# COMMAND ----------

#Use list to generate different models and compare
def different_feature_importances(feature_counts, feature_imp, dataframe):
    best_score = -1
    best_model = None
    best_count = -1
    for count in feature_counts:
        features_to_use = feature_imp[0:count]
        feature_idx = [x for x in features_to_use['idx']]
        print(feature_idx)
        slicer = VectorSlicer(inputCol="features", outputCol="features_reduced", indices=feature_idx)
        reduced_df = slicer.transform(dataframe)
        print("Reduced DF")
        rf = RandomForestClassifier(labelCol="DEP_DEL15", featuresCol="features_reduced", impurity="gini", seed = 2018)
        train_df, test_df = train_test_split(reduced_df, 0.8, 0.2, 2018)
        print("Split DF")
        cur_model = rf.fit(train_df)
        print("Fit Model")
        cur_prediction = cur_model.transform(test_df)
        print("Tested Model")
        cur_score = calculate_f1_score(cur_prediction)
        print("Using " + str(count) + " features : " + str(cur_score))
        if cur_score > best_score:
            best_score = cur_score
            best_model = cur_model
            best_count = count
        print("=================")
    print("Best result achieved using " + str(best_count) + " features : " + str(best_score)) 
    return best_model

    

# COMMAND ----------

feature_counts = [50, 30, 20, 15, 10]
best_model = different_feature_importances(feature_counts, ranked_features, samp_dataframe)
# pred_19 = best_model.transform(p_dataframe_19)
# print(calculate_f1_score(pred_19))

# COMMAND ----------

# MAGIC %md
# MAGIC #Using the new features

# COMMAND ----------

df_airlines_weather_full_updated = spark.read.parquet(f"{blob_url}/df_airlines_weather_holidays")
df_airlines_weather_full_updated.cache()
df_airlines_weather_u = df_airlines_weather_full_updated.where(df_airlines_weather_full_updated.FL_DATE_UTC < '2019-01-01')
df_airlines_weather_u.cache()
df_airlines_weather_19_u = df_airlines_weather_full_updated.where(df_airlines_weather_full_updated.FL_DATE_UTC >= '2019-01-01')
df_airlines_weather_19_u.cache()

# COMMAND ----------

def simplify_dataframe_full_u(df):
    df = df.where((df.DEP_DEL15).isNotNull())
    
    relevant_columns = ["FL_DATETIME_UTC", "DEP_HOUR_UTC", "OP_CARRIER_FL_NUM", "ORIGIN", "DEST", "DEP_HOUR", "Day_of_Week", "Month", "Distance", "number_dep_delay_origin", "number_flights_origin", "number_arr_delay_tail", "number_flights_tail", "percent_dep_delay_origin", "percent_arr_delay_tail", "isHoliday", "ELEVATION", "WIND_GUST_SPEED_RATE", "WIND_DIR_IMP", "WIND_SPEED_RATE_IMP", "VIS_DISTANCE_IMP", "CEILING_HEIGHT_IMP", "AIR_TEMP_IMP", "DEW_POINT_TEMPERATURE_IMP", "SEA_LEVEL_PRESSURE_IMP", "ATMOS_PRESSURE_3H_IMP", "PRECIPITATION_RATE", "PRESENT_ACC_OHE", "PAST_ACC_OHE", "ATMOS_PRESSURE_TENDENCY_3H_OHE", "WIND_TYPE_OHE", "DEP_DEL15"]

    simplified_dataframe = df.select([col for col in relevant_columns])
    simplified_dataframe = simplified_dataframe.groupBy(simplified_dataframe.FL_DATETIME_UTC, simplified_dataframe.DEP_HOUR_UTC, simplified_dataframe.OP_CARRIER_FL_NUM, simplified_dataframe.ORIGIN) \
                                   .agg(max("DEST").alias("DEST"), \
                                        max("DEP_HOUR").alias("DEP_HOUR"), \
                                        max("Day_of_Week").alias("Day_of_Week"),\
                                        max("Month").alias("Month"),\
                                        max("Distance").alias("Distance"), \
                                        max("number_dep_delay_origin").alias("number_dep_delay_origin"), \
                                        max("number_flights_origin").alias("number_flights_origin"),\
                                        max("number_arr_delay_tail").alias("number_arr_delay_tail"),\
                                        max("number_flights_tail").alias("number_flights_tail"),\
                                        max("percent_dep_delay_origin").alias("percent_dep_delay_origin"),\
                                        max("percent_arr_delay_tail").alias("percent_arr_delay_tail"),\
                                        max("isHoliday").alias("isHoliday"),\
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

def prepare_dataframe_full_u(df):
#     non_null_df = df.where((df.ELEVATION).isNotNull())
    feature_columns = ["DEP_HOUR", "Day_of_Week", "Month", "Distance", "number_dep_delay_origin", "number_flights_origin", "number_arr_delay_tail", "number_flights_tail", "percent_dep_delay_origin", "percent_arr_delay_tail", "isHoliday", "ELEVATION", "WIND_GUST_SPEED_RATE", "WIND_DIR_IMP", "WIND_SPEED_RATE_IMP", "VIS_DISTANCE_IMP", "CEILING_HEIGHT_IMP", "AIR_TEMP_IMP", "DEW_POINT_TEMPERATURE_IMP", "SEA_LEVEL_PRESSURE_IMP", "ATMOS_PRESSURE_3H_IMP", "PRECIPITATION_RATE", "PRESENT_ACC_OHE", "PAST_ACC_OHE", "ATMOS_PRESSURE_TENDENCY_3H_OHE", "WIND_TYPE_OHE"]
    assembler = VectorAssembler(inputCols = feature_columns, outputCol = "features").setHandleInvalid("skip")
    non_null_df = assembler.transform(df)
    return non_null_df

# COMMAND ----------

def fold_train_test_assignment(dataframe, folds, train_weight):
    max_date = dataframe.groupBy().agg(max("FL_DATETIME_UTC")).collect()[0]['max(FL_DATETIME_UTC)']
    min_date = dataframe.groupBy().agg(min("FL_DATETIME_UTC")).collect()[0]['min(FL_DATETIME_UTC)']
    prev_cutoff = min_date
    processed_df = {}
    for i in range(folds):
        cutoff_date = min_date + ((max_date - min_date) / folds) * (i+1)
        if i+1 == folds: 
            cur_fold_df = dataframe.where((dataframe.FL_DATETIME_UTC >= prev_cutoff) & (dataframe.FL_DATETIME_UTC <= cutoff_date))
        else: 
            cur_fold_df = dataframe.where((dataframe.FL_DATETIME_UTC >= prev_cutoff) & (dataframe.FL_DATETIME_UTC < cutoff_date))
        
        train_cutoff = prev_cutoff + (cutoff_date - prev_cutoff) * train_weight
        
        cur_fold_df = cur_fold_df.withColumn("cv", when(col("FL_DATETIME_UTC") <= train_cutoff, "train").otherwise("test"))
        
        processed_df["df" + str(i+1)] = cur_fold_df
        
        prev_cutoff = cutoff_date
    
    return processed_df

# COMMAND ----------

s_dataframe_u = simplify_dataframe_full_u(df_airlines_weather_u)

# COMMAND ----------

p_dataframe_u = prepare_dataframe_full_u(s_dataframe_u)

# COMMAND ----------

samp_dataframe_u = sample_dataframe(p_dataframe_u)

# COMMAND ----------

split_df = fold_train_test_assignment(samp_dataframe_u, 1, 0.8)['df1']
train_df_u = split_df.where(split_df.cv == 'train')
test_df_u = split_df.where(split_df.cv == 'test')

# COMMAND ----------

rf = RandomForestClassifier(labelCol="DEP_DEL15", featuresCol="features", impurity="gini")
model_u = rf.fit(train_df_u)
predictions_u = model_u.transform(test_df_u)
predictions_u.cache()

# COMMAND ----------

ranked_features = ExtractFeatureImp(model_u.featureImportances, split_df, "features")
display(ranked_features)

# COMMAND ----------

print(calculate_f1_score(predictions_u))

# COMMAND ----------

#Use list to generate different models and compare
def different_feature_importances_u(feature_counts, feature_imp, dataframe):
    best_score = -1
    best_model = None
    best_count = -1
    for count in feature_counts:
        features_to_use = feature_imp[0:count]
        feature_idx = [x for x in features_to_use['idx']]
        print(feature_idx)
        slicer = VectorSlicer(inputCol="features", outputCol="features_reduced", indices=feature_idx)
        reduced_df = slicer.transform(dataframe)
        print("Reduced DF")
        rf = RandomForestClassifier(labelCol="DEP_DEL15", featuresCol="features_reduced", impurity="gini", seed = 2018)
        split_df = fold_train_test_assignment(reduced_df, 1, 0.8)['df1']
        train_df = split_df.where(split_df.cv == 'train')
        test_df = split_df.where(split_df.cv == 'test')
        print("Split DF")
        cur_model = rf.fit(train_df)
        print("Fit Model")
        cur_prediction = cur_model.transform(test_df)
        print("Tested Model")
        cur_score = calculate_f1_score(cur_prediction)
        print("Using " + str(count) + " features : " + str(cur_score))
        if cur_score > best_score:
            best_score = cur_score
            best_model = cur_model
            best_count = count
        print("=================")
    print("Best result achieved using " + str(best_count) + " features : " + str(best_score)) 
    return best_model

# COMMAND ----------

feature_counts = [147, 130, 100, 70, 50, 30, 20, 15, 10, 5]
best_model = different_feature_importances_u(feature_counts, ranked_features, samp_dataframe_u)

# COMMAND ----------

