# Databricks notebook source
# DBTITLE 1,Package imports
import pyspark.sql.functions as F
from pyspark.sql.functions import col, split, isnan, when, count, concat, lit, desc, floor, max, avg, min, regexp_replace, first
from pyspark.sql.functions import hour, minute, floor, to_date, rand
from pyspark.sql.functions import array, create_map, struct
from pyspark.sql.functions import substring, lit, udf, row_number, lower
from pyspark.sql.functions import sum as ps_sum, count as ps_count
from pyspark.sql.types import IntegerType
from pyspark.sql import SQLContext
from pyspark.sql import DataFrameNaFunctions
from pyspark.sql import DataFrame
from pyspark.sql import Row

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Binarizer, MinMaxScaler, StandardScaler, Imputer, BucketedRandomProjectionLSH,VectorSlicer
from pyspark.ml.feature import OneHotEncoder, VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.linalg import Vectors,VectorUDT

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import pandas as pd
import numpy as np
import random

%matplotlib inline

import matplotlib.pyplot as plt
from matplotlib import image
import seaborn as sns
from functools import reduce

from pyspark.sql.window import *
from pyspark.sql.window import Window

# COMMAND ----------

# DBTITLE 1,File path and environment
blob_container = "team08-container" # The name of your container created in https://portal.azure.com
storage_account = "team08" # The name of your Storage account created in https://portal.azure.com
secret_scope = "team08-scope" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "team08-key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# COMMAND ----------

# DBTITLE 1,Read the joined file and extract data till end of 2018
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

# DBTITLE 1,Separate the Departure delayed ones from the non-delayed records
df_airlines_weather_delay = df_airlines_weather[df_airlines_weather["DEP_DEL15"] == 1.0]
df_airlines_weather_nodelay = df_airlines_weather[df_airlines_weather["DEP_DEL15"] == 0.0]

# COMMAND ----------

# DBTITLE 1,Print the respective counts
print(df_airlines_weather_delay.count())
print(df_airlines_weather_nodelay.count())

# COMMAND ----------

# DBTITLE 1,Extract relevant columns and fill with averages/first for non-available values
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

def prepare_dataframe_full(df, include_wt=False, wt=(0,0)):
    non_null_df = df.where((df.ELEVATION).isNotNull())
    feature_columns = ["ELEVATION", "WIND_GUST_SPEED_RATE", "WIND_DIR_IMP", "WIND_SPEED_RATE_IMP", "VIS_DISTANCE_IMP", "CEILING_HEIGHT_IMP", "AIR_TEMP_IMP", "DEW_POINT_TEMPERATURE_IMP", "SEA_LEVEL_PRESSURE_IMP", "ATMOS_PRESSURE_3H_IMP", "PRECIPITATION_RATE", "PRESENT_ACC_OHE", "PAST_ACC_OHE", "ATMOS_PRESSURE_TENDENCY_3H_OHE", "WIND_TYPE_OHE"]
    
    ############# add weight column for the delayed and non-delayed flights ########################
    if include_wt:
        if wt == (0,0):
            count_delay = non_null_df[non_null_df["DEP_DEL15"] == 1.0].count()
            count_no_delay = non_null_df[non_null_df["DEP_DEL15"] == 0.0].count()
            wt = ((count_delay + count_no_delay) / (count_delay * 2), (count_delay + count_no_delay) / (count_no_delay * 2))
            
        non_null_df = non_null_df.withColumn("WEIGHT", when(col("DEP_DEL15") == 1.0, wt[0]).otherwise(wt[1]))
        print("Weight for delayed flights:", wt[0], "|Weight for non-delayed flights:", wt[1])
    ################################################################################################
    assembler = VectorAssembler(inputCols=feature_columns, outputCol = "features")
    non_null_df = assembler.transform(non_null_df)
    return non_null_df, wt[0], wt[1]

# COMMAND ----------

# DBTITLE 1,Sample from the prepared dataframe 
def sample_dataframe(dataframe, alpha, delay_samp=0.95):
    
    #separate the rows for delayed Vs non-delayed flights into 2 dataframes
    df_delayed = dataframe.where(dataframe.DEP_DEL15 == 1.0)
    df_non_delayed = dataframe.where(dataframe.DEP_DEL15 == 0.0)
    
    # sample the delayed dataframe, by default, up to 95%
    #df_delayed_samp = df_delayed.sample(False, delay_samp, seed = 2018)
    
    # calculation for the sample ratio for the no_delay case
    delay_cnt = df_delayed.count()
    no_delay_cnt = df_non_delayed.count()
    no_delay_samp = delay_cnt / no_delay_cnt + ((no_delay_cnt - delay_cnt) / no_delay_cnt) * alpha
    
    df_non_delayed_samp = df_non_delayed.sample(False, no_delay_samp, seed=2018)
    
    return df_delayed.union(df_non_delayed_samp), delay_cnt, no_delay_cnt * no_delay_samp

# COMMAND ----------

# DBTITLE 1,SMOTE for handling data skew (not effective, long run time)
def smote(vectorized_sdf, seed=2018, knn=5, multiplier=3, bucketLen=5.0):
    
    # separate the two classes of labels
    dataInput_min = vectorized_sdf[vectorized_sdf['DEP_DEL15'] == 1]
    dataInput_maj = vectorized_sdf[vectorized_sdf['DEP_DEL15'] == 0]

    brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", seed=seed, bucketLength=bucketLen)
    model = brp.fit(dataInput_min)
    model.transform(dataInput_min)
    
    # distance is calculated from brp's param inputCol ("features" in this case) - join two datasets to approximately find 
    # all pairs of rows whose distance are smaller than the threshold (set to "inf" here). 
    self_join_w_distance = model.approxSimilarityJoin(dataInput_min, dataInput_min, float("inf"), distCol="EuclideanDistance")
    
    # remove self-comparison (distance 0)
    self_join_w_distance = self_join_w_distance.filter(self_join_w_distance.EuclideanDistance > 0)
    
    over_original_rows = Window.partitionBy("datasetA").orderBy("EuclideanDistance")
    
    self_similarity_df = self_join_w_distance.withColumn("r_num", F.row_number().over(over_original_rows))

    self_similarity_df_selected = self_similarity_df.filter(self_similarity_df.r_num <= knn)

    over_original_rows_no_order = Window.partitionBy('datasetA')
    
    # list to store batches of synthetic data
    res = []
    
    # two udf for vector add and subtract, subtraction include a random factor [0,1]
    subtract_vector_udf = F.udf(lambda arr: random.uniform(0, 1)*(arr[0]-arr[1]), VectorUDT())
    add_vector_udf = F.udf(lambda arr: arr[0]+arr[1], VectorUDT())
    
    # retain original columns
    original_cols = dataInput_min.columns
    
    for i in range(multiplier):
        print("generating batch %s of synthetic instances"%i)
        
        # logic to randomly select neighbour: pick the largest random number generated row as the neighbour
        df_random_sel = self_similarity_df_selected.withColumn("rand", F.rand()).withColumn('max_rand', F.max('rand').over(over_original_rows_no_order))\
                            .where(F.col('rand') == F.col('max_rand')).drop(*['max_rand','rand','r_num'])
        
        # create synthetic feature numerical part
        df_vec_diff = df_random_sel.select('*', subtract_vector_udf(F.array('datasetA.features', 'datasetB.features')).alias('vec_diff'))
        df_vec_modified = df_vec_diff.select('*', add_vector_udf(F.array('datasetA.features', 'vec_diff')).alias('features'))
        
        # for categorical cols, either pick original or the neighbour's cat values
        for c in original_cols:
            # randomly select neighbour or original data
            col_sub = random.choice(['datasetA', 'datasetB'])
            val = "{0}.{1}".format(col_sub,c)
            if c != 'features':
                # do not unpack original numerical features
                df_vec_modified = df_vec_modified.withColumn(c, F.col(val))
        
        # this df_vec_modified is the synthetic minority instances,
        df_vec_modified = df_vec_modified.drop(*['datasetA','datasetB','vec_diff','EuclideanDistance'])
        
        res.append(df_vec_modified)
    
    dfunion = reduce(DataFrame.unionAll, res)
    
    # union synthetic instances with original full (both minority and majority) df
    oversampled_df = dfunion.union(vectorized_sdf.select(dfunion.columns))
    
    return oversampled_df
    

# COMMAND ----------

# DBTITLE 1,Train-Test split with standard scaler for mean 0 and division by std. deviation
def train_test_split(dataframe, train_weight, test_weight, seed=2018, standardize=True):
    dates = dataframe.select(dataframe.FL_DATE_UTC).distinct().orderBy(dataframe.FL_DATE_UTC)
    
    train_dates, test_dates = dates.randomSplit([train_weight, test_weight], seed)
    train_dates = set(train_dates.toPandas()['FL_DATE_UTC'])
    test_dates = set(test_dates.toPandas()['FL_DATE_UTC'])
    
    train_df = dataframe.where(dataframe.FL_DATE_UTC.isin(train_dates))
    test_df = dataframe.where(dataframe.FL_DATE_UTC.isin(test_dates))
    
    #train_df = smote(train_df)
    
    if standardize:
        std_scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
        scaled_train = std_scaler.fit(train_df)
        scaled_train_df = scaled_train.transform(train_df)
        scaled_test_df = scaled_train.transform(test_df)
        return scaled_train_df, scaled_test_df
    else:
        return train_df, test_df

# COMMAND ----------

# DBTITLE 1,F1-Score calculation
def calculate_f1_score(dataframe, roc=False, auc=False):
    TP = dataframe.where((dataframe.DEP_DEL15 == 1) & (dataframe.prediction == 1)).count()
    FP = dataframe.where((dataframe.DEP_DEL15 == 0) & (dataframe.prediction == 1)).count()
    TN = dataframe.where((dataframe.DEP_DEL15 == 0) & (dataframe.prediction == 0)).count()
    FN = dataframe.where((dataframe.DEP_DEL15 == 1) & (dataframe.prediction == 0)).count()
    print(TP, FP, TN, FN)   
    f1 = np.divide(TP, (TP + 0.5 * (FP + FN)))
    return f1

# COMMAND ----------

# DBTITLE 1,Logistic Regression
def logreg(N, dataframe, train_weight, test_weight):
    scores = []
    random.seed(3000)
    seeds = random.sample(range(1000, 6000), N)
    
    for i in range(N):
        seed = seeds[i]
        train_df, test_df = train_test_split(dataframe, train_weight, test_weight, seed)
        lr = LogisticRegression(labelCol="DEP_DEL15", 
                                featuresCol="scaled_features", 
                                elasticNetParam=0.5, 
                                aggregationDepth=8, 
                                standardization=False,
                                threshold=0.5)
        pipeline = Pipeline(stages=[lr])
        model = pipeline.fit(train_df)
        ######################## Placeholder for CV ###############################
        #paramGrid = ParamGridBuilder() \
        #                .addGrid(lr.regParam, [0.05]) \
        #                .build()
        #crossval = CrossValidator(estimator=pipeline,
        #                  estimatorParamMaps=paramGrid,
        #                  evaluator=BinaryClassificationEvaluator(),
        #                  numFolds=5)
        #model = crossval.fit(train_df)
        #model = cvModel.bestModel
        ############################################################################
        predictions = model.transform(test_df)
        pred_labels = predictions.select(predictions.DEP_DEL15, predictions.prediction)
        f1 = calculate_f1_score(pred_labels)
        print("Iteration " + str(i+1) + ": " + str(f1))
        scores.append(f1)
    
    return np.mean(scores)

# COMMAND ----------

#split_list = [0.25] * 4
#seed = 2018
#df_airlines_weather_nodelay_splits = df_airlines_weather_nodelay.randomSplit(split_list, seed)
df_airlines_weather_nodelay_samp = df_airlines_weather_nodelay.sample(False, 0.4, seed=2018)

# COMMAND ----------

# DBTITLE 1,Sample from the entire dataframe
samp_list = [0.1, 0.2, 0.3, 0.4]
for i in range(len(samp_list)):
    df_airlines_weather_nodelay_samp = df_airlines_weather_nodelay.sample(False, samp_list[i], seed=2018)
    df_airlines_weather_scaled = df_airlines_weather_delay.union(df_airlines_weather_nodelay_samp)
    simplified_dataframe = simplify_dataframe_full(df_airlines_weather_scaled)
    prepared_dataframe = prepare_dataframe_full(simplified_dataframe)
    delay_cnt = prepared_dataframe.where(prepared_dataframe.DEP_DEL15 == 1.0).count()
    no_delay_cnt = prepared_dataframe.where(prepared_dataframe.DEP_DEL15 == 0.0).count()
    f1_score = logreg(1, prepared_dataframe, 0.725, 0.275)
    print("F1 Score: ", f1_score, "|", "Delay count: ", delay_cnt, "|", "No_delay count: ", round(no_delay_cnt), sep='')

# COMMAND ----------

# MAGIC %md
# MAGIC |F1-Score |Delay data|No_Delay data|LR Thresh|
# MAGIC |---------|----------|-------------|---------|
# MAGIC |  0.49   |  4.3M    |    5.0M     |   0.5   |
# MAGIC |  0.40   |  4.3M    |    8.9M     |   0.5   |
# MAGIC |  0.004  |  4.3M    |   12.0M     |   0.5   |
# MAGIC |  0.001  |  4.3M    |   14.4M     |   0.5   |

# COMMAND ----------

# DBTITLE 1,Sample from the prepared dataframe
alpha_list = [0.0, 0.1, 0.2, 0.3]
simplified_dataframe = simplify_dataframe_full(df_airlines_weather)
prepared_dataframe = prepare_dataframe_full(simplified_dataframe)
for i in range(len(alpha_list)):
    sampled_dataframe, delay_cnt, no_delay_cnt = sample_dataframe(prepared_dataframe, alpha_list[i])
    f1_score = logreg(1, sampled_dataframe, 0.725, 0.275)
    print("F1 Score: ", f1_score, "|", "Delay count: ", delay_cnt, "|", "No_delay count: ", round(no_delay_cnt), sep='')

# COMMAND ----------

# MAGIC %md
# MAGIC Table of F1-score where the sampling done after eliminating all rows where ELEVATION is null
# MAGIC 
# MAGIC |F1-Score |Delay data|No_Delay data|LR Thresh|
# MAGIC |---------|----------|-------------|---------|
# MAGIC |  0.57   |  4.3M    |    4.3M     |   0.5   |
# MAGIC |  0.33   |  4.3M    |    5.8M     |   0.5   |
# MAGIC |  0.15   |  4.3M    |    7.4M     |   0.5   |
# MAGIC |  0.06   |  4.3M    |    8.9M     |   0.5   |

# COMMAND ----------

# DBTITLE 1,Weight calculation in the full dataframe
simplified_dataframe = simplify_dataframe_full(df_airlines_weather)

#prepared_dataframe = prepare_dataframe_full(simplified_dataframe, include_wt=False)
#f1_score = logreg(1, prepared_dataframe, 0.8, 0.2)
#print("F1 Score: " + str(f1_score))

prepared_dataframe, wt_delay, wt_no_delay = prepare_dataframe_full(simplified_dataframe, include_wt=True)
f1_score = logreg(1, prepared_dataframe, 0.8, 0.2)
print("F1 Score: ", f1_score)

# COMMAND ----------

# DBTITLE 1,Run the best model for prediction 
s_dataframe = simplify_dataframe_full(df_airlines_weather)
s_dataframe.cache()
p_dataframe = prepare_dataframe_full(s_dataframe)
p_dataframe.cache()
samp_dataframe, delay_cnt, no_delay_cnt = sample_dataframe(p_dataframe, 0)
samp_dataframe.cache()

print("Delay count:", delay_cnt, "| ", "No_delay count:", round(no_delay_cnt))

# COMMAND ----------

# DBTITLE 1,Generate test and train splits
train_df, test_df = train_test_split(samp_dataframe, 0.725, 0.275, standardize=False)
train_df.cache()
test_df.cache()
print(train_df.count())
print(test_df.count())

# COMMAND ----------

# DBTITLE 1,Standardize the train data and apply the standardization to test
std_scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
scaled_train = std_scaler.fit(train_df)
scaled_train_df = scaled_train.transform(train_df)
scaled_test_df = scaled_train.transform(test_df)
scaled_train_df.cache()
scaled_test_df.cache()

print(scaled_train_df.count())
print(scaled_test_df.count())

# COMMAND ----------

# DBTITLE 1,Logistic Regression Model
lr = LogisticRegression(labelCol="DEP_DEL15", 
                                featuresCol="scaled_features", 
                                elasticNetParam=0.5, 
                                aggregationDepth=8, 
                                standardization=False,
                                threshold=0.5)
model = lr.fit(scaled_train_df)
predictions = model.transform(scaled_test_df)
predictions.cache()
print("TEST:", scaled_test_df.count())
print("PRED:", predictions.count())

# COMMAND ----------

print(calculate_f1_score(predictions))

# COMMAND ----------

# DBTITLE 1,Prepare the 2019 data
s_dataframe_19 = simplify_dataframe_full(df_airlines_weather_19)
s_dataframe_19.cache()
p_dataframe_19 = prepare_dataframe_full(s_dataframe_19)
p_dataframe_19.cache()

# COMMAND ----------

# DBTITLE 1,Predict on 2019 data
scaled_test_df_19 = scaled_train.transform(p_dataframe_19)

predictions_19 = model.transform(scaled_test_df_19)
predictions_19.cache()

# COMMAND ----------

print(calculate_f1_score(predictions_19))

# COMMAND ----------

# DBTITLE 1,Run LR with weighted model
sw_dataframe = simplify_dataframe_full(df_airlines_weather)
sw_dataframe.cache()

# COMMAND ----------

pw_dataframe, wt_delay, wt_no_delay = prepare_dataframe_full(sw_dataframe, include_wt=True)
pw_dataframe.cache()

# COMMAND ----------

# DBTITLE 1,Test-train split, no standardization
trainw_df, testw_df = train_test_split(pw_dataframe, 0.8, 0.2, standardize=False)
trainw_df.cache()
testw_df.cache()

# COMMAND ----------

# DBTITLE 1,Standardize test-train
std_scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
scaledw_train = std_scaler.fit(trainw_df)
scaledw_train_df = scaledw_train.transform(trainw_df)
scaledw_test_df = scaledw_train.transform(testw_df)
scaledw_train_df.cache()
scaledw_test_df.cache()

# COMMAND ----------

# DBTITLE 1,Run LR for the weighted case
lr = LogisticRegression(labelCol="DEP_DEL15", 
                                featuresCol="scaled_features", 
                                elasticNetParam=0.5, 
                                aggregationDepth=8, 
                                standardization=False,
                                threshold=0.5,
                                weightCol='WEIGHT')
modelw = lr.fit(scaledw_train_df)
predictions_w = modelw.transform(scaledw_test_df)
predictions_w.cache()

print(calculate_f1_score(predictions_w))

# COMMAND ----------

# DBTITLE 1,Run the weighted model on 2019 data
sw_dataframe_19 = simplify_dataframe_full(df_airlines_weather_19)
sw_dataframe_19
pw_dataframe_19, w0, w1 = prepare_dataframe_full(sw_dataframe_19, include_wt=True, wt=(wt_delay, wt_no_delay))
pw_dataframe_19

scaledw_test_df_19 = scaledw_train.transform(pw_dataframe_19)

predictions_19w = modelw.transform(scaledw_test_df_19)
predictions_19w.cache()

print(calculate_f1_score(predictions_19w))

# COMMAND ----------

# DBTITLE 1,Extension for ROC curve plotting.
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

# DBTITLE 1,Call the superclass to get the ROC 
preds = predictions_19w.select('DEP_DEL15','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['DEP_DEL15'])))
points = CurveMetrics(preds).get_curve('roc')

plt.figure()
x_val = [x[0] for x in points]
y_val = [x[1] for x in points]
plt.title("ROC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.plot(x_val, y_val)

# COMMAND ----------

display(df_airlines_weather)