# Databricks notebook source
from pyspark.sql.functions import max, min, when, col, avg, first
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier

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
df_airlines_weather_3m = df_airlines_weather_full.where(df_airlines_weather_full.FL_DATE_UTC <= '2015-04-01')
df_airlines_weather_3m.cache()
print(df_airlines_weather_3m.count())
print(df_airlines_weather_full.count())
print(df_airlines_weather.count())
print(df_airlines_weather_19.count())

# COMMAND ----------

def simplify_dataframe_full(df):
    df = df.where((df.DEP_DEL15).isNotNull())
    
    relevant_columns = ["FL_DATETIME_UTC", "DEP_HOUR_UTC", "OP_CARRIER_FL_NUM", "ORIGIN", "DEST", "ELEVATION", "WIND_GUST_SPEED_RATE", "WIND_DIR_IMP", "WIND_SPEED_RATE_IMP", "VIS_DISTANCE_IMP", "CEILING_HEIGHT_IMP", "AIR_TEMP_IMP", "DEW_POINT_TEMPERATURE_IMP", "SEA_LEVEL_PRESSURE_IMP", "ATMOS_PRESSURE_3H_IMP", "PRECIPITATION_RATE", "PRESENT_ACC_OHE", "PAST_ACC_OHE", "ATMOS_PRESSURE_TENDENCY_3H_OHE", "WIND_TYPE_OHE", "DEP_DEL15"]

    simplified_dataframe = df.select([col for col in relevant_columns])
    simplified_dataframe = simplified_dataframe.groupBy(simplified_dataframe.FL_DATETIME_UTC, simplified_dataframe.DEP_HOUR_UTC, simplified_dataframe.OP_CARRIER_FL_NUM, simplified_dataframe.ORIGIN) \
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

def sample_dataframe(dataframe): 
    delayed_df = dataframe.where(dataframe.DEP_DEL15 == 1)
    non_delayed_df = dataframe.where(dataframe.DEP_DEL15 == 0)
    delayed_df_samp = delayed_df.sample(False, 0.90, seed = 2018)
    non_delayed_df_samp = non_delayed_df.sample(False, 0.20, seed = 2018)
    return delayed_df_samp.union(non_delayed_df_samp)

# COMMAND ----------

s_dataframe = simplify_dataframe_full(df_airlines_weather)
p_dataframe = prepare_dataframe_full(s_dataframe)
samp_dataframe = sample_dataframe(p_dataframe)
dfs = fold_train_test_assignment(samp_dataframe, 5, 0.8)

# COMMAND ----------

ls /dbfs/

# COMMAND ----------

spark.sparkContext.addPyFile("dbfs:/custom_cv.py")

# COMMAND ----------

from custom_cv import CustomCrossValidator

# COMMAND ----------

rf = RandomForestClassifier(labelCol="DEP_DEL15", featuresCol="features")
grid = ParamGridBuilder()\
        .addGrid(rf.maxDepth, [25,27,30])\
        .addGrid(rf.numTrees, [15, 20, 25])\
        .addGrid(rf.minInstancesPerNode, [30, 50, 100])\
        .build()
evaluator = BinaryClassificationEvaluator(labelCol="DEP_DEL15")

# COMMAND ----------

cv = CustomCrossValidator(estimator=rf, estimatorParamMaps=grid, evaluator=evaluator, splitWord = ('train', 'test'), cvCol = 'cv', parallelism=4)
best_model = cv.fit(dfs)

# COMMAND ----------

rf = RandomForestClassifier(labelCol="DEP_DEL15", featuresCol="features")
grid = ParamGridBuilder()\
        .addGrid(rf.maxDepth, [26,27,28])\
        .addGrid(rf.numTrees, [19, 20, 21])\
        .addGrid(rf.minInstancesPerNode, [40, 50, 60])\
        .build()
evaluator = BinaryClassificationEvaluator(labelCol="DEP_DEL15")

# COMMAND ----------

cv = CustomCrossValidator(estimator=rf, estimatorParamMaps=grid, evaluator=evaluator, splitWord = ('train', 'test'), cvCol = 'cv', parallelism=4)
best_model = cv.fit(dfs)

# COMMAND ----------

rf = RandomForestClassifier(labelCol="DEP_DEL15", featuresCol="features")
grid = ParamGridBuilder()\
        .addGrid(rf.maxDepth, [26])\
        .addGrid(rf.numTrees, [21, 22, 23, 24])\
        .addGrid(rf.minInstancesPerNode, [35, 40, 45])\
        .build()
evaluator = BinaryClassificationEvaluator(labelCol="DEP_DEL15")

# COMMAND ----------

cv = CustomCrossValidator(estimator=rf, estimatorParamMaps=grid, evaluator=evaluator, splitWord = ('train', 'test'), cvCol = 'cv', parallelism=4)
best_model = cv.fit(dfs)

# COMMAND ----------

rf = RandomForestClassifier(labelCol="DEP_DEL15", featuresCol="features")
grid = ParamGridBuilder()\
        .addGrid(rf.maxDepth, [26])\
        .addGrid(rf.numTrees, [22])\
        .addGrid(rf.minInstancesPerNode, [35, 40, 45])\
        .build()
evaluator = BinaryClassificationEvaluator(labelCol="DEP_DEL15")