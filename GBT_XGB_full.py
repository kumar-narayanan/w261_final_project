# Databricks notebook source
# DBTITLE 1,Imports
from pyspark.sql.functions import col,split,isnan, when, count, concat, lit, desc, floor, max, avg, min, regexp_replace, first
import numpy as np
import pandas as pd
import random
from pyspark.sql.types import IntegerType, array
from pyspark.sql import SQLContext
from pyspark.sql import DataFrameNaFunctions
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import Binarizer
from pyspark.ml.feature import OneHotEncoder, VectorAssembler, StringIndexer, VectorIndexer, VectorSlicer

# COMMAND ----------

# DBTITLE 1,Environment
blob_container = "team08-container" # The name of your container created in https://portal.azure.com
storage_account = "team08" # The name of your Storage account created in https://portal.azure.com
secret_scope = "team08-scope" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "team08-key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# COMMAND ----------

# MAGIC %md
# MAGIC #Using the new features

# COMMAND ----------

# DBTITLE 1,Read the data
df_airlines_weather_full_updated = spark.read.parquet(f"{blob_url}/df_airlines_weather_holidays")
df_airlines_weather_full_updated.cache()
df_airlines_weather_u = df_airlines_weather_full_updated.where(df_airlines_weather_full_updated.FL_DATE_UTC < '2019-01-01')
df_airlines_weather_u.cache()
df_airlines_weather_19_u = df_airlines_weather_full_updated.where(df_airlines_weather_full_updated.FL_DATE_UTC >= '2019-01-01')
df_airlines_weather_19_u.cache()

# COMMAND ----------

# DBTITLE 1,Data frame with the relevant columns and fill in empty values
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

# DBTITLE 1,Vectorization with option to add weights for data imbalance
def prepare_dataframe_full_u(df, include_wt=False, wt=(0,0)):
#     non_null_df = df.where((df.ELEVATION).isNotNull())
    feature_columns = ["DEP_HOUR", "Day_of_Week", "Month", "Distance", "number_dep_delay_origin", "number_flights_origin", "number_arr_delay_tail", "number_flights_tail", "percent_dep_delay_origin", "percent_arr_delay_tail", "isHoliday", "ELEVATION", "WIND_GUST_SPEED_RATE", "WIND_DIR_IMP", "WIND_SPEED_RATE_IMP", "VIS_DISTANCE_IMP", "CEILING_HEIGHT_IMP", "AIR_TEMP_IMP", "DEW_POINT_TEMPERATURE_IMP", "SEA_LEVEL_PRESSURE_IMP", "ATMOS_PRESSURE_3H_IMP", "PRECIPITATION_RATE", "PRESENT_ACC_OHE", "PAST_ACC_OHE", "ATMOS_PRESSURE_TENDENCY_3H_OHE", "WIND_TYPE_OHE"]
    assembler = VectorAssembler(inputCols = feature_columns, outputCol = "features").setHandleInvalid("skip")
    non_null_df = assembler.transform(df)
    
    ############# add weight column for the delayed and non-delayed flights ########################
    if include_wt:
        if wt == (0,0):
            count_delay = non_null_df[non_null_df["DEP_DEL15"] == 1.0].count()
            count_no_delay = non_null_df[non_null_df["DEP_DEL15"] == 0.0].count()
            wt = ((count_delay + count_no_delay) / (count_delay * 2), (count_delay + count_no_delay) / (count_no_delay * 2))
            
        non_null_df = non_null_df.withColumn("WEIGHT", when(col("DEP_DEL15") == 1.0, wt[0]).otherwise(wt[1]))
        print("Weight for delayed flights:", wt[0], "|Weight for non-delayed flights:", wt[1])
    ################################################################################################
    return non_null_df, wt[0], wt[1]

# COMMAND ----------

# DBTITLE 1,Generate sub-data-frames equal to folds by splitting the data frame
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

# DBTITLE 1,Data frame for model to compensate for imbalance through under sampling of majority class
def sample_dataframe(dataframe): 
    delayed_df = dataframe.where(dataframe.DEP_DEL15 == 1)
    non_delayed_df = dataframe.where(dataframe.DEP_DEL15 == 0)
    delayed_df_samp = delayed_df.sample(False, 0.90, seed = 2018)
    non_delayed_df_samp = non_delayed_df.sample(False, 0.20, seed = 2018)
    return delayed_df_samp.union(non_delayed_df_samp)

# COMMAND ----------

# DBTITLE 1,Extract important features by looking at the score against each feature
def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))

# COMMAND ----------

# DBTITLE 1,F1 score calculation
def calculate_f1_score(dataframe):
    TP = dataframe.where((dataframe.DEP_DEL15 == 1) & (dataframe.prediction == 1)).count()
    FP = dataframe.where((dataframe.DEP_DEL15 == 0) & (dataframe.prediction == 1)).count()
    TN = dataframe.where((dataframe.DEP_DEL15 == 0) & (dataframe.prediction == 0)).count()
    FN = dataframe.where((dataframe.DEP_DEL15 == 1) & (dataframe.prediction == 0)).count()
    print(TP, FP, TN, FN)
    f1 = np.divide(TP, (TP + 0.5*(FP+FN)))
    return f1

# COMMAND ----------

# DBTITLE 1,Generate data with relevant columns
s_dataframe_u = simplify_dataframe_full_u(df_airlines_weather_u)

# COMMAND ----------

# DBTITLE 1,Vectorize the column
p_dataframe_u, _, _ = prepare_dataframe_full_u(s_dataframe_u)

# COMMAND ----------

# DBTITLE 1,Sample to compensate for imbalanced data
samp_dataframe_u = sample_dataframe(p_dataframe_u)

# COMMAND ----------

# DBTITLE 1,Train-Test split
split_df = fold_train_test_assignment(samp_dataframe_u, 1, 0.8)['df1']
train_df_u = split_df.where(split_df.cv == 'train')
test_df_u = split_df.where(split_df.cv == 'test')

# COMMAND ----------

# DBTITLE 1,Run Gradient Based Tree Model
gbt = GBTClassifier(labelCol="DEP_DEL15", featuresCol="features", seed=2018)
model_u = gbt.fit(train_df_u)
predictions_u = model_u.transform(test_df_u)
predictions_u.cache()

# COMMAND ----------

# DBTITLE 1,Get the feature rank
ranked_features_gbt = ExtractFeatureImp(model_u.featureImportances, split_df, "features")
display(ranked_features_gbt)

# COMMAND ----------

# DBTITLE 1,Get F1 score
print(calculate_f1_score(predictions_u))

# COMMAND ----------

display(predictions_u)

# COMMAND ----------

# DBTITLE 1,Comparison of different models
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
        rf = GBTClassifier(labelCol="DEP_DEL15", featuresCol="features_reduced", seed = 2018)
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

# DBTITLE 1,Get the best model by taking different subset of features starting from full feature set
feature_counts_gbt = [147, 50, 30, 20, 15, 10, 5]
best_model_gbt = different_feature_importances_u(feature_counts_gbt, ranked_features_gbt, samp_dataframe_u)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC #### Error Analysis for top 10 features 

# COMMAND ----------

# DBTITLE 1,Get the top 10 features, vectorize and run GBT
features_to_use = ranked_features_gbt[0:10]
feature_idx = [x for x in features_to_use['idx']]

print(feature_idx)

slicer = VectorSlicer(inputCol="features", outputCol="features_reduced", indices=feature_idx)
reduced_df = slicer.transform(samp_dataframe_u)

gbt_ea = GBTClassifier(labelCol="DEP_DEL15", featuresCol="features_reduced", seed = 2018)

split_df = fold_train_test_assignment(reduced_df, 1, 0.8)['df1']

train_df = split_df.where(split_df.cv == 'train')
test_df = split_df.where(split_df.cv == 'test')

model_ea = gbt_ea.fit(train_df)
prediction_ea = model_ea.transform(test_df)
score_ea = calculate_f1_score(prediction_ea)

# COMMAND ----------

display(prediction_ea)

# COMMAND ----------

# DBTITLE 1,Select relevant columns from the predicted frame
df = prediction_ea.select("FL_DATETIME_UTC", "OP_CARRIER_FL_NUM", "number_arr_delay_tail", "percent_arr_delay_tail", "percent_dep_delay_origin", "DEP_HOUR", "Distance", "AIR_TEMP_IMP", "ELEVATION", "Month", "number_flights_origin", "number_dep_delay_origin", "probability", "prediction", "features_reduced", "DEP_DEL15")

# COMMAND ----------

display(df)

# COMMAND ----------

# True Positives
df = df.withColumn("cm_label", 
                      when((df.prediction == "1") & (df.DEP_DEL15 == "1"),"TP")
                      .when((df.prediction == "0") & (df.DEP_DEL15 == "0"),"TN")
                      .when((df.prediction == "0") & (df.DEP_DEL15 == "1"),"FN")
                      .otherwise("FP")
                  )

# COMMAND ----------

display(df.where(df.cm_label == "FP"))

# COMMAND ----------

display(df.where(df.cm_label == "FN"))

# COMMAND ----------

display(df.where(df.cm_label == "TP"))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### GBT with weighting

# COMMAND ----------

# DBTITLE 1,Get simplified data frame with features and add weights inversely proportional to minority and majority class
s_dataframe_u = simplify_dataframe_full_u(df_airlines_weather_u)
pw_dataframe_u, wt_delay, wt_no_delay = prepare_dataframe_full_u(s_dataframe_u, include_wt=True)

# COMMAND ----------

# DBTITLE 1,Sample weighted data frame
sampw_dataframe_u = sample_dataframe(pw_dataframe_u)

# COMMAND ----------

# DBTITLE 1,Train-test split
splitw_df = fold_train_test_assignment(sampw_dataframe_u, 1, 0.8)['df1']
trainw_df_u = splitw_df.where(splitw_df.cv == 'train')
testw_df_u = splitw_df.where(splitw_df.cv == 'test')

# COMMAND ----------

# DBTITLE 1,Gradient boost model
from sparkdl.xgboost import XgboostClassifier

#xgb = XgboostClassifier(num_workers=6, labelCol="DEP_DEL15", max_depth=5, weightCol="WEIGHT", missing=0.0)
xgb = XgboostClassifier(num_workers=6, labelCol="DEP_DEL15", max_depth=5, missing=0.0)
model_xgb_u = xgb.fit(trainw_df_u)
predictions_xgb_u = model_xgb_u.transform(testw_df_u)
predictions_xgb_u.cache()

# COMMAND ----------

# DBTITLE 1,F1 scores
print(calculate_f1_score(predictions_xgb_u))

# COMMAND ----------

# DBTITLE 1,Function to get feature importance for XGB
def different_feature_importances_xgb_u(feature_counts, feature_imp, dataframe):
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
        xgb = XgboostClassifier(labelCol="DEP_DEL15", featuresCol="features_reduced", seed = 2018)
        split_df = fold_train_test_assignment(reduced_df, 1, 0.8)['df1']
        train_df = split_df.where(split_df.cv == 'train')
        test_df = split_df.where(split_df.cv == 'test')
        print("Split DF")
        cur_model = xgb.fit(train_df)
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

# DBTITLE 1,XGB for various over sub-features starting from the full set
feature_counts_xgb = [147, 50, 30, 20, 15, 10, 5]
best_model_xgb = different_feature_importances_u(feature_counts_xgb, ranked_features, sampw_dataframe_u)

# COMMAND ----------

# DBTITLE 1,Extract the relevant features and vectorize 2019 data set 
s_dataframe_19_u = simplify_dataframe_full_u(df_airlines_weather_19_u)
pw_dataframe_19_u, wt_delay, wt_no_delay = prepare_dataframe_full_u(s_dataframe_19_u)


# COMMAND ----------

# DBTITLE 1,Sample the 2019 data
sampw_dataframe_19_u = sample_dataframe(pw_dataframe_19_u)

# COMMAND ----------

# DBTITLE 1,Predictions on 2019 data set
# predictions_xgb_19_u = model_xgb_u.transform(sampw_dataframe_19_u)
predictions_xgb_19_u = model_xgb_u.transform(pw_dataframe_19_u)

# COMMAND ----------

predictions_xgb_19_u.write.mode("overwrite").parquet(f"{blob_url}/predictions/xgb")

# COMMAND ----------

# DBTITLE 1,F1 score
print(calculate_f1_score(predictions_xgb_19_u))

# COMMAND ----------

display(predictions_xgb_19_u)

# COMMAND ----------

# DBTITLE 1,Super class for ROC an PR curve generation
from pyspark.mllib.evaluation import BinaryClassificationMetrics

class CurveMetrics(BinaryClassificationMetrics):
    def __init__(self, *args):
        super(CurveMetrics, self).__init__(*args)

    def _to_list(self, rdd):
        points = []
        # Note this collect could be inefficient for large datasets 
        # considering there may be one probability per datapoint (at most)
        # The Scala version takes a numBins parameter, 
        # but it doesn't seem possible to pass this from Python to Java
        for row in rdd.collect():
            # Results are returned as type scala.Tuple2, 
            # which doesn't appear to have a py4j mapping
            points += [(float(row._1()), float(row._2()))]
        return points

    def get_curve(self, method):
        rdd = getattr(self._java_model, method)().toJavaRDD()
        return self._to_list(rdd)

# COMMAND ----------

# DBTITLE 1,Data frame with label, probability and predictions
preds = predictions_xgb_19_u.select('DEP_DEL15','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['DEP_DEL15'])))

# COMMAND ----------

# DBTITLE 1,Points for ROC
points = CurveMetrics(preds).get_curve('roc')

# COMMAND ----------

# DBTITLE 1,Plot the curve.
# MAGIC %matplotlib inline
# MAGIC 
# MAGIC import matplotlib.pyplot as plt
# MAGIC from matplotlib import image
# MAGIC import seaborn as sns
# MAGIC 
# MAGIC plt.figure()
# MAGIC x_val = [x[0] for x in points]
# MAGIC y_val = [x[1] for x in points]
# MAGIC plt.title("ROC")
# MAGIC plt.xlabel("False Positive Rate")
# MAGIC plt.ylabel("True Positive Rate")
# MAGIC plt.plot(x_val, y_val)

# COMMAND ----------

