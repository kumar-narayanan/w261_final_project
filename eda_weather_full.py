# Databricks notebook source
# MAGIC %md
# MAGIC ## EDA for Full Weather Data

# COMMAND ----------

from pyspark.sql.functions import col,split,isnan, when, count
from pyspark.sql.functions import hour, minute, floor, to_date, lit
from pyspark.sql.types import IntegerType
import pandas as pd

blob_container = "team08-container" # The name of your container created in https://portal.azure.com
storage_account = "team08" # The name of your Storage account created in https://portal.azure.com
secret_scope = "team08-scope" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "team08-key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# COMMAND ----------

# Inspect the Mount's Final Project folder 
display(dbutils.fs.ls("/mnt/mids-w261/datasets_final_project"))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Common Functions

# COMMAND ----------

def drop_columns(df, col_names):
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

def null_values_eda(df):
  total_count = df.count()
  for column in df.columns:
    count = df.filter(df[column].isNull()).count()
    print(column, ":", count, ":", total_count, ":", count * 100 / total_count)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 1: Load full raw weather data

# COMMAND ----------

df_weather = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*")

# COMMAND ----------

df_weather.printSchema()

# COMMAND ----------

display(df_weather)

# COMMAND ----------

display(df_weather.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 2: Data cleaning and Feature Extraction

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC #### Split Weather Feature Columns into multiple Features

# COMMAND ----------

df_weather = split_columns(df_weather)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Drop Columns / Features
# MAGIC 
# MAGIC Drop the additional weather columns as well as the base columns from which we extracted features in prior step.

# COMMAND ----------

drop_additional_weather_columns = {"AW1","GA1","GA2","GA3","GA4","GE1","GF1","KA2","MA1","OD1","OD2","REM","EQD","AW2","AX4","GD1","AW5","GN1","AJ1","AW3","MK1","KA4","GG3","AN1","RH1","AU5","HL1","OB1","AT8","AW7","AZ1","CH1","RH3","GK1","IB1","AX1","CT1","AK1","CN2","OE1","MW5","AO1","KA3","AA3","CR1","CF2","KB2","GM1","AT5","MW6","MG1","AH6","AU2","GD2","AW4","MF1","AH2","AH3","OE3","AT6","AL2","AL3","AX5","IB2","AI3","CV3","WA1","GH1","KF1","CU2","CT3","SA1","AU1","KD2","AI5","GO1","GD3","CG3","AI1","AL1","AW6","MW4","AX6","CV1","ME1","KC2","CN1","UA1","GD5","UG2","AT3","AT4","GJ1","MV1","GA5","CT2","CG2","ED1","AE1","CO1","KE1","KB1","AI4","MW3","KG2","AA2","AX2","RH2","OE2","CU3","MH1","AM1","AU4","GA6","KG1","AU3","AT7","KD1","GL1","IA1","GG2","OD3","UG1","CB1","AI6","CI1","CV2","AZ2","AD1","AH1","WD1","AA4","KC1","IA2","CF3","AI2","AT1","GD4","AX3","AH4","KB3","CU1","CN4","AT2","CG1","CF1","GG1","MV2","CW1","GG4","AB1","AH5","CN3","AY2","KA1", "SLP","DEW","TMP","VIS","CIG","WND","MD1","MW1","MW2","OC1","AA1","AY1","LATITUDE","LONGITUDE","SOURCE","REPORT_TYPE","QUALITY_CONTROL","NAME","CALL_SIGN","AIR_TEMP_QUALITY_CODE","DEW_POINT_QUALITY_CODE","SEA_LEVEL_PRESSURE_QUALITY_CODE","WIND_DIR_QUALITY_CODE","WIND_SPEED_QUALITY_CODE","VIS_DISTANCE_QUALITY_CODE","CEILING_QUALITY_CODE","CEILING_DETERM_CODE","CAVOK","VIS_DISTANCE_QUALITY_CODE","VIS_QUALITY_VARIABILITY_CODE","VIS_VARIABILTY_CODE"}

df_weather = drop_columns(df_weather, drop_additional_weather_columns)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Drop Rows
# MAGIC 
# MAGIC Drop rows with missing STATION

# COMMAND ----------

def drop_missing_station_records(df):
  return df.na.drop(subset=["STATION"])

df_weather = drop_missing_station_records(df_weather)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Convert Data Types

# COMMAND ----------

df_weather = df_weather \
                              .withColumn("WIND_DIR",df_weather.WIND_DIR.cast(IntegerType())) \
                              .withColumn("WIND_SPEED_RATE",df_weather.WIND_SPEED_RATE.cast(IntegerType())) \
                              .withColumn("CEILING_HEIGHT",df_weather.CEILING_HEIGHT.cast(IntegerType())) \
                              .withColumn("VIS_DISTANCE",df_weather.VIS_DISTANCE.cast(IntegerType())) \
                              .withColumn("AIR_TEMP",df_weather.AIR_TEMP.cast(IntegerType())) \
                              .withColumn("DEW_POINT_TEMPERATURE",df_weather.DEW_POINT_TEMPERATURE.cast(IntegerType())) \
                              .withColumn("SEA_LEVEL_PRESSURE",df_weather.SEA_LEVEL_PRESSURE.cast(IntegerType())) \
                              .withColumn("ATMOS_PRESSURE_3H",df_weather.ATMOS_PRESSURE_3H.cast(IntegerType()))

# COMMAND ----------

df_weather.printSchema()

# COMMAND ----------

# MAGIC %md ### Correlation Matrix

# COMMAND ----------

from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

# convert to vector column first
vector_col = "corr_features"
inputColumns = ["WIND_DIR", "WIND_SPEED_RATE", "CEILING_HEIGHT", "VIS_DISTANCE", "AIR_TEMP", "DEW_POINT_TEMPERATURE", "SEA_LEVEL_PRESSURE"]
assembler = VectorAssembler(inputCols=inputColumns, outputCol=vector_col)
df_vector = assembler.transform(df_weather).select(vector_col)

# get correlation matrix
matrix = Correlation.corr(df_vector, vector_col).collect()[0][0]

# COMMAND ----------

import seaborn as sns

corrmatrix = matrix.toArray().tolist()
cmap = sns.light_palette("#4e909e", as_cmap=True)
plt.title("Correlation Matrix", fontsize =20)
sns.heatmap(corrmatrix, 
        xticklabels=inputColumns,
        yticklabels=inputColumns, cmap=cmap, annot=True, vmin=-1, vmax=1)

# COMMAND ----------

display(df_weather.select("WIND_DIR", "WIND_SPEED_RATE", "CEILING_HEIGHT", "VIS_DISTANCE", "AIR_TEMP", "DEW_POINT_TEMPERATURE", "SEA_LEVEL_PRESSURE"))

# COMMAND ----------

display(df_weather.where(df_weather.WIND_DIR != 999).select("WIND_DIR"))

# COMMAND ----------

display(df_weather.where(df_weather.WIND_SPEED_RATE != 9999).select("WIND_SPEED_RATE").sample(fraction=0.1, seed=2018))

# COMMAND ----------

display(df_weather.where(df_weather.CEILING_HEIGHT != 99999).select("CEILING_HEIGHT"))

# COMMAND ----------

display(df_weather.where(df_weather.VIS_DISTANCE != 999999).select("VIS_DISTANCE"))

# COMMAND ----------

display(df_weather.where(df_weather.AIR_TEMP != 9999).select("AIR_TEMP"))

# COMMAND ----------

display(df_weather.select("DEW_POINT_TEMPERATURE").where(df_weather.DEW_POINT_TEMPERATURE != 9999))

# COMMAND ----------

display(df_weather.select("SEA_LEVEL_PRESSURE").where(df_weather.SEA_LEVEL_PRESSURE != 99999))

# COMMAND ----------

display(df_weather.select("WIND_DIR", "WIND_SPEED_RATE", "CEILING_HEIGHT", "VIS_DISTANCE", "AIR_TEMP", "DEW_POINT_TEMPERATURE", "SEA_LEVEL_PRESSURE").summary())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Feature Average Imputation

# COMMAND ----------

# MAGIC %md Any missing value for a weather measurement is indicated by the highest value in the range of values for the feature. For example, any missing air temperature is given by 9999. So, we replace these missing values with null.

# COMMAND ----------

feature_mcode_mapping = {"WIND_DIR":999, "WIND_SPEED_RATE":9999, "VIS_DISTANCE":999999,"CEILING_HEIGHT":99999, "AIR_TEMP":9999, "DEW_POINT_TEMPERATURE":9999, "SEA_LEVEL_PRESSURE":99999, "ATMOS_PRESSURE_3H":999 }

def replace(column, value):
  '''This function replaces missing values with null'''
  return when(column != value, column).otherwise(lit(None))

for feature, missing_code in feature_mcode_mapping.items():
    df_weather = df_weather.withColumn(feature, replace(col(feature), missing_code))

# COMMAND ----------

def apply_avg_imputation(df, col_name, missing_code):
  '''This function computes the average(mean) of the last 720 records at the station for a given feature'''
  window = Window.partitionBy("STATION").orderBy("DATE").rowsBetween(-720, 0)
  condition = when( (col(col_name).isNull()) | (col(col_name) == missing_code), floor(mean(col(col_name)).over(window))).otherwise(col(col_name))
  return df.withColumn(col_name+"_IMP", condition)     

# COMMAND ----------

# MAGIC %md Apply average of last 720 records for any weather readings that are missing

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql.functions import avg,mean

feature_mcode_mapping = {"WIND_DIR":999, "WIND_SPEED_RATE":9999, "VIS_DISTANCE":999999,"CEILING_HEIGHT":99999, "AIR_TEMP":9999, "DEW_POINT_TEMPERATURE":9999, "SEA_LEVEL_PRESSURE":99999, "ATMOS_PRESSURE_3H":999 }

for column, missing_code in feature_mcode_mapping.items():
  df_weather=apply_avg_imputation(df_weather, column, missing_code)

# COMMAND ----------

# MAGIC %md create new imputed features which is the average weather reading

# COMMAND ----------

def apply_avg_imputation_across_feature(df, feature_name):
  window = Window.partitionBy()
  condition = when(col(feature_name).isNull(), floor(mean(col(feature_name)).over(window))).otherwise(col(feature_name))
  return df.withColumn(feature_name, condition)      

for feature, missing_code in feature_mcode_mapping.items():
  df_weather=apply_avg_imputation_across_feature(df_weather, feature+"_IMP")

# COMMAND ----------

from pyspark.sql.functions import col,when
df_weather = df_weather \
                                .withColumn("PRECIPITATION_PERIOD_QTY", when(col("PRECIPITATION_PERIOD_QTY")=="" ,"99").otherwise(col("PRECIPITATION_PERIOD_QTY"))) \
                                .withColumn("WIND_GUST_SPEED_RATE", when(col("WIND_GUST_SPEED_RATE")=="" ,"0").otherwise(col("WIND_GUST_SPEED_RATE"))) \
                                .withColumn("WIND_TYPE", when(col("WIND_SPEED_RATE")=="0000" ,"C").otherwise(col("WIND_TYPE"))) \
                                .withColumn("PRESENT_ATMOSPHERIC_CONDITION_CODE", when(col("PRESENT_ATMOSPHERIC_CONDITION_CODE")=="" ,"NA").otherwise(col("PRESENT_ATMOSPHERIC_CONDITION_CODE"))) \
                                .withColumn("PAST_ATMOSPHERIC_CONDITION_CODE", when(col("PAST_ATMOSPHERIC_CONDITION_CODE")=="" ,"NA").otherwise(col("PAST_ATMOSPHERIC_CONDITION_CODE"))) \
                                .withColumn("ATMOS_PRESSURE_TENDENCY_3H", when(col("ATMOS_PRESSURE_TENDENCY_3H")=="" ,"NA").otherwise(col("ATMOS_PRESSURE_TENDENCY_3H"))) \
                                .withColumn("WIND_TYPE", when(col("WIND_TYPE")=="" ,"NA").otherwise(col("WIND_TYPE")))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Cast Data Types for remaining features

# COMMAND ----------

df_weather = df_weather \
                              .withColumn("WIND_GUST_SPEED_RATE",df_weather.WIND_GUST_SPEED_RATE.cast(IntegerType())) \
                              .withColumn("PRECIPITATION_DEPTH",df_weather.PRECIPITATION_DEPTH.cast(IntegerType())) \
                              .withColumn("PRECIPITATION_PERIOD_QTY",df_weather.PRECIPITATION_PERIOD_QTY.cast(IntegerType())) \
                              .withColumn("PAST_ATMOSPHERIC_CONDITION_CODE_DURATION",df_weather.PAST_ATMOSPHERIC_CONDITION_CODE_DURATION.cast(IntegerType()))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Fill Missing Values for remaining features

# COMMAND ----------

def fill_missing_values_99(df):
  return df.na.fill(value='99',subset=["PAST_ATMOSPHERIC_CONDITION_CODE_DURATION"])   

df_weather = fill_missing_values_99(df_weather)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Derive Precipitation Rate Per Hour
# MAGIC 
# MAGIC *PRECIPITATION_RATE = PRECIPITATION_DEPTH / PRECIPITATION_PERIOD_QTY*

# COMMAND ----------

df_weather = df_weather.na.replace(99, 0, subset = ["PRECIPITATION_PERIOD_QTY"]) \
                       .na.replace(9999, 0, subset = ["PRECIPITATION_DEPTH"]) 

df_weather = df_weather.withColumn("PRECIPITATION_RATE", df_weather["PRECIPITATION_DEPTH"] / df_weather["PRECIPITATION_PERIOD_QTY"]).na.fill(0, subset=["PRECIPITATION_RATE"]) 
df_weather = drop_columns(df_weather, ["PRECIPITATION_PERIOD_QTY", "PRECIPITATION_DEPTH"])

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Feature Encoding

# COMMAND ----------

from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder

indexer = StringIndexer(inputCols=["PRESENT_ATMOSPHERIC_CONDITION_CODE","PAST_ATMOSPHERIC_CONDITION_CODE","ATMOS_PRESSURE_TENDENCY_3H","WIND_TYPE"], outputCols=["PRESENT_ACC_INDEX","PAST_ACC_INDEX","ATMOS_PRESSURE_TENDENCY_3H_INDEX","WIND_TYPE_INDEX"])
df_weather = indexer.fit(df_weather).transform(df_weather)

ohe = OneHotEncoder(inputCols=["PRESENT_ACC_INDEX","PAST_ACC_INDEX","ATMOS_PRESSURE_TENDENCY_3H_INDEX","WIND_TYPE_INDEX"], outputCols=["PRESENT_ACC_OHE","PAST_ACC_OHE","ATMOS_PRESSURE_TENDENCY_3H_OHE","WIND_TYPE_OHE"])
df_weather=ohe.fit(df_weather).transform(df_weather)

# Drop base columns since we have OHE columns now, as well as the index columns since they are not needed
cols = {"PRESENT_ATMOSPHERIC_CONDITION_CODE","PAST_ATMOSPHERIC_CONDITION_CODE","ATMOS_PRESSURE_TENDENCY_3H","WIND_TYPE","PRESENT_ACC_INDEX","PAST_ACC_INDEX","ATMOS_PRESSURE_TENDENCY_3H_INDEX","WIND_TYPE_INDEX"}
df_weather=drop_columns(df_weather,cols)

# COMMAND ----------

cols = { "WIND_DIR", "WIND_SPEED_RATE", "VIS_DISTANCE","CEILING_HEIGHT","AIR_TEMP", "DEW_POINT_TEMPERATURE", "SEA_LEVEL_PRESSURE", "ATMOS_PRESSURE_3H"}
df_weather=drop_columns(df_weather,cols)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 3: Review Final Weather DataFrame

# COMMAND ----------

display(df_weather)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Summary Statistics After Feature and Data Processing

# COMMAND ----------

summary = df_weather.summary()

# COMMAND ----------

display(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 4: Write data to blob storage in Parquet format

# COMMAND ----------

df_weather.write.parquet(f"{blob_url}/df_weather")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Appendix

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Appendix A : List of Features from Weather Data Set
# MAGIC 
# MAGIC | Variable | DataType | Data Source | Missing % |
# MAGIC |---|---|---|---|
# MAGIC |STATION||Available in Weather Dataset|0.61%|
# MAGIC |DATE||Available in Weather Dataset|0%||
# MAGIC |ELEVATION||Available in Weather Dataset|0%|
# MAGIC |WIND_DIR||Available in Weather Dataset|0%|
# MAGIC |WIND_TYPE||Available in Weather Dataset|16.23%|
# MAGIC |WIND_SPEED_RATE||Available in Weather Dataset| 0% |
# MAGIC |CELING_HEIGHT||Available in Weather Dataset|48.0349%|
# MAGIC |VIS_DISTANCE||Available in Weather Dataset|32.35%|
# MAGIC |AIR_TEMP||Available in Weather Dataset|1.5%|
# MAGIC |DEW_POINT_TEMP||Available in Weather Dataset|15.42% |
# MAGIC |SEA_LEVEL_PRESSURE||Available in Weather Dataset|65.14%|
# MAGIC 
# MAGIC ### Additional Attributes
# MAGIC 
# MAGIC **MW1**
# MAGIC PRESENT-WEATHER-OBSERVATION **Detailed Codes**
# MAGIC 
# MAGIC | Variable | DataType | Data Source |
# MAGIC |---|---|---|
# MAGIC |PRESENT_ATMOSPHERIC_CONDITION_CODE||MW1|
# MAGIC 
# MAGIC **AY1, AY2**
# MAGIC Past Weather Observations - contains indicators for fog, drizzle, rain, showers. 
# MAGIC 
# MAGIC | Variable | DataType | Data Source |
# MAGIC |---|---|---|
# MAGIC |PAST_ATMOSPHERIC_CONDITION_CODE||AY1|
# MAGIC |PAST_ATMOSPHERIC_CONDITION_CODE_PERIOD||AY1|
# MAGIC 
# MAGIC **MD1**
# MAGIC ATMOSPHERIC-PRESSURE-CHANGE
# MAGIC Captures the tendancy of atmospheric pressure - is it increasing, decreasing, stable in comparison to last 3 hours and 24 hours.
# MAGIC 
# MAGIC | Variable | DataType | Data Source |
# MAGIC |---|---|---|
# MAGIC |ATMOS_PRESSURE_TENDENCY_CODE||MD1|
# MAGIC |ATMOS_PRESSURE_3H||MD1|
# MAGIC 
# MAGIC **OC1**
# MAGIC WIND-GUST-OBSERVATION 
# MAGIC 
# MAGIC | Variable | DataType | Data Source |
# MAGIC |---|---|---|
# MAGIC |WIND_GUST_SPEED_RATE||OC1|
# MAGIC 
# MAGIC **AA1, AA2**
# MAGIC PRECIPITATION Depth (amount of prec measured)
# MAGIC 
# MAGIC | Variable | DataType | Data Source |
# MAGIC |---|---|---|
# MAGIC |PRECIPITATION_PERIOD_QTY||AA1|
# MAGIC |PRECIPITATION_DEPTH||AA1|

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Appendix B : Missing Values Table
# MAGIC 
# MAGIC |Base Feature|Missing Value|
# MAGIC |---|---|
# MAGIC STATION||
# MAGIC DATE||
# MAGIC ELEVATION | 9999 = missing|
# MAGIC WIND_DIR | 999 = Missing|
# MAGIC WIND_TYPE | 9 = Missing|
# MAGIC WIND_SPEED_RATE | 9999 = Missing|
# MAGIC CEILING_HEIGHT | 99999 = Missing|
# MAGIC VIS_DISTANCE | 999999 = Missing|
# MAGIC AIR_TEMP | 9999 = Missing|
# MAGIC DEW_POINT_TEMPERATURE | 9999 = Missing|
# MAGIC SEA_LEVEL_PRESSURE | 99999 = Missing|
# MAGIC PRESENT_ATMOSPHERIC_CONDITION_CODE ||
# MAGIC PAST_ATMOSPHERIC_CONDITION_CODE||
# MAGIC PAST_ATMOSPHERIC_CONDITION_CODE_DURATION | 99 = Missing|
# MAGIC ATMOS_PRESSURE_TENDENCY_3H | 9 = Missing|
# MAGIC ATMOS_PRESSURE_3H | 999 = Missing|
# MAGIC WIND_GUST_SPEED_RATE | 9999 = Missing|
# MAGIC PRECIPITATION_PERIOD_QTY | 99 = Missing|
# MAGIC PRECIPITATION_DEPTH | 9999 = Missing|
# MAGIC Hour | |
# MAGIC Date_Only ||