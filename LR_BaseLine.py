# Databricks notebook source
from pyspark.sql.functions import col,split,isnan, when, count, concat, lit, desc, floor, max, avg, min, regexp_replace,first
from pyspark.sql.functions import col, max
from pyspark.sql.types import IntegerType
from pyspark.sql import SQLContext
from pyspark.sql import DataFrameNaFunctions

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Binarizer, MinMaxScaler, StandardScaler, Imputer
from pyspark.ml.feature import OneHotEncoder, VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from sklearn.metrics import f1_score

import pandas as pd
import numpy as np
import random

# COMMAND ----------

blob_container = "team08-container" # The name of your container created in https://portal.azure.com
storage_account = "team08" # The name of your Storage account created in https://portal.azure.com
secret_scope = "team08-scope" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "team08-key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# COMMAND ----------

#df_weather_airline_3m = spark.read.parquet(f"{blob_url}/df_airlines_weather_3m")
df_weather_airline_6m = spark.read.parquet(f"{blob_url}/df_airlines_weather_6m")

# COMMAND ----------

def simplify_dataframe(dataframe):

    #missing WIND_TYPE
    relevant_columns = ["FL_DATE_UTC", "DEP_HOUR_UTC", "OP_CARRIER_FL_NUM", "ORIGIN", "DEST", "WIND_DIR", "WIND_TYPE", "WIND_SPEED_RATE", "CEILING_HEIGHT", "VIS_DISTANCE", "AIR_TEMP", "DEW_POINT_TEMPERATURE", "SEA_LEVEL_PRESSURE", "DEP_DEL15"]

    converted_df = dataframe.withColumn("WIND_DIR", dataframe["WIND_DIR"].cast(IntegerType())) \
                            .withColumn("WIND_SPEED_RATE", dataframe["WIND_SPEED_RATE"].cast(IntegerType())) \
                            .withColumn("CEILING_HEIGHT", dataframe["CEILING_HEIGHT"].cast(IntegerType())) \
                            .withColumn("VIS_DISTANCE", dataframe["VIS_DISTANCE"].cast(IntegerType())) \
                            .withColumn("AIR_TEMP", dataframe["AIR_TEMP"].cast(IntegerType())) \
                            .withColumn("DEW_POINT_TEMPERATURE", dataframe["DEW_POINT_TEMPERATURE"].cast(IntegerType()))\
                            .withColumn("SEA_LEVEL_PRESSURE", dataframe["SEA_LEVEL_PRESSURE"].cast(IntegerType()))
    
    cleaned_df = converted_df.na.replace(999, 0, subset=["WIND_DIR"])\
                             .na.fill(0, subset=["WIND_DIR"])\
                             .na.replace(99999, 0, subset=["CEILING_HEIGHT"])\
                             .na.replace(999999, 0, subset=["VIS_DISTANCE"])\
                             .na.fill(0, subset=["VIS_DISTANCE"])\
                             .na.replace(9999, 0, subset=["AIR_TEMP"])\
                             .na.replace(9999, 0, subset=["DEW_POINT_TEMPERATURE"]) \
                             .na.replace(9999, 0, subset=["SEA_LEVEL_PRESSURE"])
    
    #simple_dataframe = cleaned_df.select([col for col in relevant_columns])
    simple_dataframe = cleaned_df.select(relevant_columns)
    
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
    
    ##################### New Code ##########################
    std_scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
    scaled_train = std_scaler.fit(train_df)
    scaled_train_df = scaled_train.transform(train_df)
    scaled_test_df = scaled_train.transform(test_df)
    
    return scaled_train_df, scaled_test_df
    ################## End New Code #########################
    
    #return train_df, test_df

# COMMAND ----------

def calculate_f1_score(dataframe):
    TP = dataframe.where((dataframe.DEP_DEL15 == 1) & (dataframe.prediction == 1)).count()
    FP = dataframe.where((dataframe.DEP_DEL15 == 0) & (dataframe.prediction == 1)).count()
    TN = dataframe.where((dataframe.DEP_DEL15 == 0) & (dataframe.prediction == 0)).count()
    FN = dataframe.where((dataframe.DEP_DEL15 == 1) & (dataframe.prediction == 0)).count()
    print(TP, FP, TN, FN)
    f1 = np.multiply(2, np.divide(TP, (TP + 0.5*(FP+FN)) ))
    return f1

# COMMAND ----------

def logreg(N, dataframe, train_weight, test_weight):
    scores = []
    random.seed(3000)
    seeds = random.sample(range(1000, 6000), N)
    
    for i in range(N):
        seed = seeds[i]
        train_df, test_df = train_test_split(dataframe, train_weight, test_weight, seed)
        lr = LogisticRegression(labelCol="DEP_DEL15", featuresCol="scaled_features", elasticNetParam=1.0, aggregationDepth=8)
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

simplified_dataframe_3m = simplify_dataframe(df_weather_airline_3m)
prepared_dataframe_3m = prepare_dataframe(simplified_dataframe_3m)
f1_score_3m = logreg(5, prepared_dataframe_3m, 0.725, 0.275)
print("3-month F1 Score: " + str(f1_score_3m))

# COMMAND ----------

simplified_dataframe_6m = simplify_dataframe(df_weather_airline_6m)
prepared_dataframe_6m = prepare_dataframe(simplified_dataframe_6m)
f1_score_6m = logreg(1, prepared_dataframe_6m, 0.75, 0.25)
print("6-month F1 Score: " + str(f1_score_6m))

# COMMAND ----------

df_airline = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data/*")

# COMMAND ----------

display(df_airline.select("Origin").distinct())

# COMMAND ----------

from pyspark.sql.functions import col,split,isnan, when, count
from pyspark.sql.functions import hour, minute, floor, to_date
from pyspark.sql.types import IntegerType

blob_container = "team08-container" # The name of your container created in https://portal.azure.com
storage_account = "team08" # The name of your Storage account created in https://portal.azure.com
secret_scope = "team08-scope" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "team08-key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# Load the 2015 Q1 for Weather
def read_weather_data(cut_off_date):
    filter_date = cut_off_date + "T00:00:00.000"
    df_weather = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*").filter(col('DATE') < filter_date)
    return df_weather

# Function for dropping columns
def drop_columns(df, col_names):
    #df = df.drop(*col_names)
    return df.drop(*col_names)

# COMMAND ----------

def split_columns(df):
    
    wind_split_col = split(df['WND'], ',')
    cig_split_col = split(df['CIG'], ',')
    vis_split_col = split(df['VIS'], ',')
    tmp_split_col = split(df['TMP'], ',')
    dew_split_col = split(df['DEW'], ',')
    slp_split_col = split(df['SLP'], ',')
    ## Additional Feature splits
    mw1_split_col = split(df['MW1'], ',')
    ay1_split_col = split(df['AY1'], ',')
    md1_split_col = split(df['MD1'], ',')
    oc1_split_col = split(df['OC1'], ',')
    aa1_split_col = split(df['AA1'], ',')

    df = df.withColumn("WIND_DIR",wind_split_col.getItem(0)) \
           .withColumn("WIND_DIR_QUALITY_CODE",wind_split_col.getItem(1)) \
           .withColumn("WIND_TYPE",wind_split_col.getItem(2)) \
           .withColumn("WIND_SPEED_RATE",wind_split_col.getItem(3)) \
           .withColumn("WIND_SPEED_QUALITY_CODE",wind_split_col.getItem(4)) \
           .withColumn("CEILING_HEIGHT",cig_split_col.getItem(0)) \
           .withColumn("CEILING_QUALITY_CODE",cig_split_col.getItem(1)) \
           .withColumn("CEILING_DETERM_CODE",cig_split_col.getItem(2)) \
           .withColumn("CAVOK",cig_split_col.getItem(3)) \
           .withColumn("VIS_DISTANCE",vis_split_col.getItem(0)) \
           .withColumn("VIS_DISTANCE_QUALITY_CODE",vis_split_col.getItem(1)) \
           .withColumn("VIS_VARIABILTY_CODE",vis_split_col.getItem(2)) \
           .withColumn("VIS_QUALITY_VARIABILITY_CODE",vis_split_col.getItem(3)) \
           .withColumn("AIR_TEMP",tmp_split_col.getItem(0)) \
           .withColumn("AIR_TEMP_QUALITY_CODE",tmp_split_col.getItem(1)) \
           .withColumn("DEW_POINT_TEMPERATURE",dew_split_col.getItem(0)) \
           .withColumn("DEW_POINT_QUALITY_CODE",dew_split_col.getItem(1)) \
           .withColumn("SEA_LEVEL_PRESSURE", slp_split_col.getItem(0)) \
           .withColumn("SEA_LEVEL_PRESSURE_QUALITY_CODE", slp_split_col.getItem(1)) \
           .withColumn("PRESENT_ATMOSPHERIC_CONDITION_CODE", mw1_split_col.getItem(0)) \
           .withColumn("PAST_ATMOSPHERIC_CONDITION_CODE", ay1_split_col.getItem(0)) \
           .withColumn("PAST_ATMOSPHERIC_CONDITION_CODE_DURATION", ay1_split_col.getItem(2)) \
           .withColumn("ATMOS_PRESSURE_TENDENCY_3H", md1_split_col.getItem(0)) \
           .withColumn("ATMOS_PRESSURE_3H", md1_split_col.getItem(2)) \
           .withColumn("WIND_GUST_SPEED_RATE", oc1_split_col.getItem(0)) \
           .withColumn("PRECIPITATION_PERIOD_QTY", aa1_split_col.getItem(0)) \
           .withColumn("PRECIPITATION_DEPTH", aa1_split_col.getItem(2)) 
    
    df = df.withColumn("Hour", floor(hour(col('DATE'))+minute(col('DATE'))/60)) \
           .withColumn("Date_Only", to_date(col('DATE'),"yyyy-MM-dd"))
    
    return df

# COMMAND ----------

df_weather_3m = read_weather_data("2015-04-01")

drop_additional_weather_columns = {"AW1","GA1","GA2","GA3","GA4","GE1","GF1","KA2","MA1","OD1","OD2","REM","EQD","AW2","AX4","GD1","AW5","GN1","AJ1","AW3","MK1","KA4","GG3","AN1","RH1","AU5","HL1","OB1","AT8","AW7","AZ1","CH1","RH3","GK1","IB1","AX1","CT1","AK1","CN2","OE1","MW5","AO1","KA3","AA3","CR1","CF2","KB2","GM1","AT5","MW6","MG1","AH6","AU2","GD2","AW4","MF1","AH2","AH3","OE3","AT6","AL2","AL3","AX5","IB2","AI3","CV3","WA1","GH1","KF1","CU2","CT3","SA1","AU1","KD2","AI5","GO1","GD3","CG3","AI1","AL1","AW6","MW4","AX6","CV1","ME1","KC2","CN1","UA1","GD5","UG2","AT3","AT4","GJ1","MV1","GA5","CT2","CG2","ED1","AE1","CO1","KE1","KB1","AI4","MW3","KG2","AA2","AX2","RH2","OE2","CU3","MH1","AM1","AU4","GA6","KG1","AU3","AT7","KD1","GL1","IA1","GG2","OD3","UG1","CB1","AI6","CI1","CV2","AZ2","AD1","AH1","WD1","AA4","KC1","IA2","CF3","AI2","AT1","GD4","AX3","AH4","KB3","CU1","CN4","AT2","CG1","CF1","GG1","MV2","CW1","GG4","AB1","AH5","CN3", "AY2", "KA1"}

df_weather_3m = drop_columns(df_weather_3m, drop_additional_weather_columns)

df_weather_3m = split_columns(df_weather_3m)

# COMMAND ----------

display(df_weather_3m)

# COMMAND ----------

import datetime

def get_col_avg(station_id, date_na, avg_col_name, avg_ignore_val, days_delta=15):
    """In the df look for STATION with station_id between the start_date and end_date.
    Filter out rows which have avg_ignore_val. Get the average"""
    
    start_date = date_na - datetime.timedelta(days=days_delta)
    end_date = date_na + datetime.timedelta(days=days_delta)
    
    col_mean = df_weather_3m.where((col('STATION') == station_id) & (col(avg_col_name) != '99999') & \
                            (col("Date_Only") >= a) & (col("Date_Only") <= b)).select(mean(col(avg_col_name))).collect()[0][0]

    return round(col_mean)

def fill_avg_value(df, col_names_dict):
    """ The col_names_dict has the form {'col_name1: 'val1', 'col_name2': 'val2'...}. The function checks if the 
    col_name has the corresponding value, in which case it'll call a function to compute average over a 4-weeks 
    period (+/- 15 days window) on either side of the Date_Only value. Returns the df with the replaced values"""
    
    for col_name, value in col_names_dict.items():
        df = df.withColumn(col_name, when(df[col_name] == value, get_col_avg(df, df["STATION"], df["Date_Only"], col_name, value)) \
               .otherwise(df[col_name]))
    return df

# COMMAND ----------

## Function for replacing missing values with the average of the feature over the previous month for the given station
## VIS_DIST, CEILING_HEIGHT, AIR_TEMP, DEW_POINT_TEMP, WIND_DIR

col_names_dict = {'CEILING_HEIGHT': '99999'}
df = fill_avg_value(df_weather_3m, col_names_dict)

# COMMAND ----------

a = datetime.datetime.strptime('2015-01-01', '%Y-%m-%d').date()
b = datetime.datetime.strptime('2015-01-15', '%Y-%m-%d').date()
value = '99999'
col_mean = df_weather_3m.filter((df_weather_3m.STATION == '3809099999')).where((col("Date_Only") >= a) & (col("Date_Only") <= b)) \
                        .filter((df_weather_3m.CEILING_HEIGHT != value)).agg({'CEILING_HEIGHT': 'avg'}).collect()


print(round(col_mean[0][0])

# COMMAND ----------

round(col_mean[0][0])

# COMMAND ----------

from pyspark.sql.functions import mean
a = datetime.datetime.strptime('2015-01-01', '%Y-%m-%d').date()
b = datetime.datetime.strptime('2015-01-15', '%Y-%m-%d').date()

df_mean = df_weather_3m.where((col('STATION') == '3809099999') & (col('CEILING_HEIGHT') != '99999') & \
                            (col("Date_Only") >= a) & (col("Date_Only") <= b)).select(mean(col('CEILING_HEIGHT'))).collect()[0][0]

#display(df_mean)

# COMMAND ----------

print(df_mean)