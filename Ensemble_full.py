# Databricks notebook source
# MAGIC %md # Ensemble Classification

# COMMAND ----------

from pyspark.sql.functions import col,split,isnan, when, count, concat, lit, desc, floor, max, avg, min, regexp_replace, first
import numpy as np
import pandas as pd
import seaborn as sn
import random
from pyspark.sql.types import IntegerType, array, FloatType
from pyspark.sql import SQLContext
from pyspark.sql import DataFrameNaFunctions
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.ml.feature import Binarizer, OneHotEncoder, VectorAssembler, StringIndexer, VectorIndexer, VectorSlicer

# COMMAND ----------

blob_container = "team08-container" # The name of your container created in https://portal.azure.com
storage_account = "team08" # The name of your Storage account created in https://portal.azure.com
secret_scope = "team08-scope" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "team08-key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# COMMAND ----------

spark.sparkContext.addPyFile("dbfs:/custom_cv.py")

# COMMAND ----------

from custom_cv import CustomCrossValidator

# COMMAND ----------

# MAGIC %md ## Load the Data

# COMMAND ----------

df_airlines_weather_full = spark.read.parquet(f"{blob_url}/df_airlines_weather_holidays")
df_airlines_weather_full.cache()

df_airlines_weather = df_airlines_weather_full.where(df_airlines_weather_full.FL_DATE_UTC < '2019-01-01')
df_airlines_weather.cache()

df_airlines_weather_19 = df_airlines_weather_full.where(df_airlines_weather_full.FL_DATE_UTC >= '2019-01-01')
df_airlines_weather_19.cache()

print(df_airlines_weather_full.count())
print(df_airlines_weather.count())
print(df_airlines_weather_19.count())

# COMMAND ----------

# MAGIC %md ## Stages

# COMMAND ----------

# MAGIC %md ### Simplify Dataframe

# COMMAND ----------

class DataFrameSimplifier(Transformer):
    """
    A custom Transformer which selects the relevant features from the airlines weather dataframe and computes the average weather.
    """

    def _transform(self, df):
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
        
        simplified_dataframe.cache()
        
        return simplified_dataframe


# COMMAND ----------

# MAGIC %md ### Downsample the dataframe

# COMMAND ----------

class DownSampler(Transformer):
    """
    A custom Transformer which down samples the the airlines weather dataframe.
    """

    def _transform(self, df):
        delayed_df = df.where(df.DEP_DEL15 == 1)
        non_delayed_df = df.where(df.DEP_DEL15 == 0)
        delayed_df_samp = delayed_df.sample(False, 0.90, seed = 2018)
        non_delayed_df_samp = non_delayed_df.sample(False, 0.20, seed = 2018)
        
        combined_df = delayed_df_samp.union(non_delayed_df_samp)
        combined_df.cache()
        return combined_df

# COMMAND ----------

# MAGIC %md ### Split into train and test

# COMMAND ----------

class TrainTestSplitter(Transformer):
    "A custom transformer which splita an airline-weather dataframe into train and test"
    
    def __init__(self, folds, train_weight):
        super(TrainTestSplitter, self).__init__()
        self.folds = folds
        self.train_weight = train_weight    
    
    def _transform(self, dataframe):
        max_date = dataframe.groupBy().agg(max("FL_DATETIME_UTC")).collect()[0]['max(FL_DATETIME_UTC)']
        min_date = dataframe.groupBy().agg(min("FL_DATETIME_UTC")).collect()[0]['min(FL_DATETIME_UTC)']
        prev_cutoff = min_date
        processed_df = {}
        for i in range(self.folds):
            cutoff_date = min_date + ((max_date - min_date) / self.folds) * (i+1)
            if i+1 == self.folds:
                cur_fold_df = dataframe.where((dataframe.FL_DATETIME_UTC >= prev_cutoff) & (dataframe.FL_DATETIME_UTC <= cutoff_date))
            else:
                cur_fold_df = dataframe.where((dataframe.FL_DATETIME_UTC >= prev_cutoff) & (dataframe.FL_DATETIME_UTC < cutoff_date))
            
            train_cutoff = prev_cutoff + (cutoff_date - prev_cutoff) * self.train_weight
            cur_fold_df = cur_fold_df.withColumn("cv", when(col("FL_DATETIME_UTC") <= train_cutoff, "train").otherwise("test"))
            processed_df["df" + str(i+1)] = cur_fold_df
            prev_cutoff = cutoff_date
        return processed_df

# COMMAND ----------

# MAGIC %md ## Create Pipeline

# COMMAND ----------

dataframe_simplifier = DataFrameSimplifier()
down_sampler = DownSampler()

feature_columns = ["DEP_HOUR", "Day_of_Week", "Month", "Distance", "number_dep_delay_origin", "number_flights_origin", "number_arr_delay_tail", "number_flights_tail", "percent_dep_delay_origin", "percent_arr_delay_tail", "isHoliday", "ELEVATION", "WIND_GUST_SPEED_RATE", "WIND_DIR_IMP", "WIND_SPEED_RATE_IMP", "VIS_DISTANCE_IMP", "CEILING_HEIGHT_IMP", "AIR_TEMP_IMP", "DEW_POINT_TEMPERATURE_IMP", "SEA_LEVEL_PRESSURE_IMP", "ATMOS_PRESSURE_3H_IMP", "PRECIPITATION_RATE", "PRESENT_ACC_OHE", "PAST_ACC_OHE", "ATMOS_PRESSURE_TENDENCY_3H_OHE", "WIND_TYPE_OHE"]
assembler = VectorAssembler(inputCols = feature_columns, outputCol = "features").setHandleInvalid("skip")

train_test_splitter = TrainTestSplitter(5, 0.8)

evaluator = BinaryClassificationEvaluator(labelCol="DEP_DEL15")
    
pipeline = Pipeline(stages=[dataframe_simplifier, assembler, down_sampler, train_test_splitter])

pipeline_model = pipeline.fit(df_airlines_weather)
dfs = pipeline_model.transform(df_airlines_weather)

# COMMAND ----------

# MAGIC %md ### Random Forest

# COMMAND ----------

random_forest = RandomForestClassifier(labelCol="DEP_DEL15", featuresCol="features")
grid = ParamGridBuilder()\
        .addGrid(random_forest.maxDepth, [25,27,30])\
        .addGrid(random_forest.numTrees, [15, 20, 25])\
        .addGrid(random_forest.minInstancesPerNode, [30, 50, 100])\
        .build()

cv = CustomCrossValidator(estimator=random_forest, estimatorParamMaps=grid, evaluator=evaluator, splitWord = ('train', 'test'), cvCol = 'cv', parallelism=4)
best_model = cv.fit(dfs)

# COMMAND ----------

# MAGIC %md ### Logistic Regression

# COMMAND ----------

lr = LogisticRegression(labelCol="DEP_DEL15", featuresCol="features")
lr_grid = ParamGridBuilder() \
               .addGrid(lr.maxIter, [0, 1])\
               .build()

# COMMAND ----------

cv_lr = CustomCrossValidator(estimator=lr, estimatorParamMaps=lr_grid, evaluator=evaluator, splitWord = ('train', 'test'), cvCol = 'cv', parallelism=4)
best_model_lr = cv_lr.fit(dfs)

# COMMAND ----------

# MAGIC %md ### Gradient Boost

# COMMAND ----------

gbt = GBTClassifier(labelCol="DEP_DEL15", featuresCol="features")
gbt_grid = ParamGridBuilder() \
               .addGrid(gbt.maxIter, [1, 2, 5])\
               .addGrid(gbt.maxDepth, [1, 1, 2])\
               .build()

# COMMAND ----------

cv_gbt = CustomCrossValidator(estimator=gbt, estimatorParamMaps=gbt_grid, evaluator=evaluator, splitWord = ('train', 'test'), cvCol = 'cv', parallelism=4)
best_model_gbt = cv_gbt.fit(dfs)

# COMMAND ----------

# MAGIC %md ## Evaluate Models

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

# MAGIC %matplotlib inline
# MAGIC 
# MAGIC import matplotlib.pyplot as plt
# MAGIC from matplotlib import image
# MAGIC import seaborn as sns
# MAGIC 
# MAGIC def plot_roc(predictions, model_name):
# MAGIC     points = CurveMetrics(predictions).get_curve('roc')
# MAGIC     
# MAGIC     plt.figure()
# MAGIC     x_val = [x[0] for x in points]
# MAGIC     y_val = [x[1] for x in points]
# MAGIC     
# MAGIC     plt.title(f"ROC for {model_name}")
# MAGIC     plt.xlabel("False Positive Rate")
# MAGIC     plt.ylabel("True Positive Rate")
# MAGIC     plt.plot(x_val, y_val)
# MAGIC     
# MAGIC def plot_pr(predictions, model_name):
# MAGIC     points = CurveMetrics(predictions).get_curve('pr')
# MAGIC     
# MAGIC     plt.figure()
# MAGIC     x_val = [x[0] for x in points]
# MAGIC     y_val = [x[1] for x in points]
# MAGIC     
# MAGIC     plt.title(f"Precision-Recall for {model_name}")
# MAGIC     plt.xlabel("Precision")
# MAGIC     plt.ylabel("Recall")
# MAGIC     plt.plot(x_val, y_val)

# COMMAND ----------

def plot_confusion_matrix(predictions, model_name):
    preds_and_labels = predictions.select(['prediction','DEP_DEL15']).withColumn('label', col('DEP_DEL15').cast(FloatType())).orderBy('prediction').select(['prediction','label'])
    metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
    
    cm = metrics.confusionMatrix().toArray()
    df_cm = pd.DataFrame(cm, index = ["True Not Delayed", "True Delayed"],
              columns = ["Predicted Not Delayed", "Predicted Delayed"])
    
    ax = plt.axes()
    sns.heatmap(df_cm, annot=True,  ax = ax, fmt='.0f', vmin=100000, vmax=3000000)
    ax.set_title(f'{model_name}')
    for t in ax.texts:
        t.set_text('{:,d}'.format(int(t.get_text())))
    plt.show()

# COMMAND ----------

test_pipeline = Pipeline(stages=[dataframe_simplifier, assembler])
test_pipeline_model = test_pipeline.fit(df_airlines_weather_19)


# COMMAND ----------

df_test = test_pipeline_model.transform(df_airlines_weather_19)

# COMMAND ----------

predictions_lr = best_model_lr.transform(df_test)

# COMMAND ----------

preds = predictions_lr.select('DEP_DEL15','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['DEP_DEL15'])))

# COMMAND ----------

plot_roc(preds, "Logistic Regression")

# COMMAND ----------

plot_pr(preds, "Logistic Regression")

# COMMAND ----------

print(calculate_f1_score(predictions_lr))

# COMMAND ----------

predictions_gbt = best_model_gbt.transform(df_test)

# COMMAND ----------

preds_gbt = predictions_gbt.select('DEP_DEL15','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['DEP_DEL15'])))

# COMMAND ----------

plot_roc(preds_gbt, "Gradient Boost")

# COMMAND ----------

plot_pr(preds_gbt, "Gradient Boost")

# COMMAND ----------

print(calculate_f1_score(predictions_gbt))

# COMMAND ----------

predictions_rf = best_model.transform(df_test)

# COMMAND ----------

preds_rf = predictions_rf.select('DEP_DEL15','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['DEP_DEL15'])))

# COMMAND ----------

plot_roc(preds_rf, "Random Forest")

# COMMAND ----------

plot_pr(preds_rf, "Random Forest")

# COMMAND ----------

print(calculate_f1_score(predictions_rf))

# COMMAND ----------

# MAGIC %md ## Save the pipeline

# COMMAND ----------

# MAGIC %md ### Training Data

# COMMAND ----------

for key, df in dfs.items():
    df.write.mode("overwrite").parquet(f"{blob_url}/pipeline/training_data/{key}")

# COMMAND ----------

# MAGIC %md ### Test Data

# COMMAND ----------

df_test.write.mode("overwrite").parquet(f"{blob_url}/pipeline/test_data")

# COMMAND ----------

# MAGIC %md ### Models

# COMMAND ----------

best_model.write().save(f"{blob_url}/models/random_forest")

# COMMAND ----------

best_model_lr.write().save(f"{blob_url}/models/logistic_regression")

# COMMAND ----------

best_model_gbt.write().save(f"{blob_url}/models/gradient_boost")

# COMMAND ----------

# MAGIC %md ### Predictions

# COMMAND ----------

def rename_prediction_columns(df, model_name):
    df = df.withColumnRenamed("rawPrediction", f"rawPrediction_{model_name}") \
            .withColumnRenamed("probability", f"probability_{model_name}") \
            .withColumnRenamed("prediction", f"prediction_{model_name}")
    return df

# COMMAND ----------

def append_predictions(dfs, model, model_name):
    predictions_dfs = {}
    for key, df in dfs.items():
        df_with_predictions = model.transform(df)
        df_with_predictions = rename_prediction_columns(df_with_predictions, model_name)
        predictions_dfs[key] = df_with_predictions
    return predictions_dfs

predictions_dfs = append_predictions(dfs, best_model, "rf")
predictions_dfs = append_predictions(predictions_dfs, best_model_lr, "lr")
predictions_dfs = append_predictions(predictions_dfs, best_model_gbt, "gbt")

# COMMAND ----------

display(predictions_dfs["df1"])

# COMMAND ----------

for key, df in predictions_dfs.items():
    df.write.mode("overwrite").parquet(f"{blob_url}/predictions/training_data/{key}")

# COMMAND ----------

predictions_rf = best_model.transform(df_test)
predictions_test = rename_prediction_columns(predictions_rf, "rf")

predictions_test = best_model_lr.transform(predictions_test)
predictions_test = rename_prediction_columns(predictions_test, "lr")

predictions_test = best_model_gbt.transform(predictions_test)
predictions_test = rename_prediction_columns(predictions_test, "gbt")

# COMMAND ----------

display(predictions_test)

# COMMAND ----------

predictions_test.write.mode("overwrite").parquet(f"{blob_url}/predictions/test_data")

# COMMAND ----------

# MAGIC %md ## Stacking Ensemble

# COMMAND ----------

level1_feature_columns = ["rawPrediction_rf", "rawPrediction_lr", "rawPrediction_gbt"]
predictions_assembler = VectorAssembler(inputCols = level1_feature_columns, outputCol = "level1_features").setHandleInvalid("skip")

transformed_predictions_dfs = {}
for key, df in predictions_dfs.items():
    transformed_predictions_dfs[key] = predictions_assembler.transform(df)

# COMMAND ----------

# MAGIC %md ### XGBoost

# COMMAND ----------

from sparkdl.xgboost import XgboostClassifier

xgbt = XgboostClassifier(labelCol="DEP_DEL15", featuresCol="level1_features", missing=0.0)

grid_xgbt = ParamGridBuilder()\
        .addGrid(xgbt.max_depth, [5,10,15])\
        .build()

cv_xgbt = CustomCrossValidator(estimator=xgbt, estimatorParamMaps=grid_xgbt, evaluator=evaluator, splitWord = ('train', 'test'), cvCol = 'cv', parallelism=4)
best_model_xgbt = cv_xgbt.fit(transformed_predictions_dfs)

# COMMAND ----------

df_transformed_predictions_test = predictions_assembler.transform(predictions_test)

# COMMAND ----------

predictions_xbgt = best_model_xgbt.transform(df_transformed_predictions_test)

# COMMAND ----------

preds_xbgt = predictions_xbgt.select('DEP_DEL15','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['DEP_DEL15'])))

# COMMAND ----------

plot_roc(preds_xbgt, "Stacking Ensembel with XGBoost")

# COMMAND ----------

plot_pr(preds_xbgt, "Stacking Ensembel with XGBoost")

# COMMAND ----------

print(calculate_f1_score(predictions_xbgt))

# COMMAND ----------

plot_confusion_matrix(predictions_xbgt, "Stacking Ensemble with XGBoost")

# COMMAND ----------

# MAGIC %md ### Logistic Regression

# COMMAND ----------

lr_ensemble = LogisticRegression(labelCol="DEP_DEL15", featuresCol="level1_features")

grid_lr_ensemble = ParamGridBuilder() \
               .addGrid(lr_ensemble.maxIter, [0, 1])\
               .build()


# COMMAND ----------

cv_lr_ensemble = CustomCrossValidator(estimator=lr_ensemble, estimatorParamMaps=grid_lr_ensemble, evaluator=evaluator, splitWord = ('train', 'test'), cvCol = 'cv', parallelism=4)
best_model_lr_ensemble = cv_lr_ensemble.fit(transformed_predictions_dfs)

# COMMAND ----------

predictions_lr_ensemble = best_model_lr_ensemble.transform(df_transformed_predictions_test)

# COMMAND ----------

display(predictions_lr_ensemble)

# COMMAND ----------

preds_lr_ensemble = predictions_lr_ensemble.select('DEP_DEL15','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['DEP_DEL15'])))

# COMMAND ----------

plot_roc(preds_lr_ensemble, "Stacking Ensemble with Logistic Regression")

# COMMAND ----------

plot_pr(preds_lr_ensemble, "Stacking Ensemble with Logistic Regression")

# COMMAND ----------

print(calculate_f1_score(predictions_lr_ensemble))

# COMMAND ----------

plot_confusion_matrix(predictions_lr_ensemble, "Stacking Ensemble with Logistic Regression")

# COMMAND ----------

# MAGIC %md ### Save the models

# COMMAND ----------

best_model_xgbt.write().save(f"{blob_url}/models/xgbt_ensemble")

# COMMAND ----------

best_model_lr_ensemble.write().save(f"{blob_url}/models/lr_ensemble")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ### Save the Transformed Data

# COMMAND ----------

for key, df in transformed_predictions_dfs.items():
    df.write.mode("overwrite").parquet(f"{blob_url}/level1/transformed_training/{key}")

# COMMAND ----------

df_transformed_predictions_test.write.mode("overwrite").parquet(f"{blob_url}/level1/transformed_test/{key}")

# COMMAND ----------

# MAGIC %md ## Model Debugging

# COMMAND ----------

errors_df = predictions_lr_ensemble.where(col('prediction') != col('DEP_DEL15'))

# COMMAND ----------

display(errors_df.limit(100))

# COMMAND ----------

errors_df.count()

# COMMAND ----------

# MAGIC %md ## Load Saved Models

# COMMAND ----------

from pyspark.ml.tuning import CrossValidatorModel
best_model_lr_ensemble = CrossValidatorModel.load(f"{blob_url}/models/lr_ensemble")

# COMMAND ----------

evaluator = BinaryClassificationEvaluator(labelCol="DEP_DEL15")

# COMMAND ----------

df_transformed_predictions_test = spark.read.parquet(f"{blob_url}/level1/transformed_test/df5")

# COMMAND ----------

predictions_lr_ensemble = best_model_lr_ensemble.transform(df_transformed_predictions_test)

# COMMAND ----------

#Output AUC
print("AUC: " + str(evaluator.evaluate(predictions_lr_ensemble)))

# COMMAND ----------

plot_confusion_matrix(predictions_lr_ensemble, "Ensemble with Logistic Regression")

# COMMAND ----------

predictions_lr_ensemble.write.mode("overwrite").parquet(f"{blob_url}/predictions/ensemble")

# COMMAND ----------

