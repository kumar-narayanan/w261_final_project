# Databricks notebook source
# MAGIC %md
# MAGIC ## EDA for Weather Data

# COMMAND ----------

from pyspark.sql.functions import col,split,isnan, when, count
from pyspark.sql.functions import hour, minute, floor, to_date,lit
from pyspark.sql.types import IntegerType

blob_container = "team08-container" # The name of your container created in https://portal.azure.com
storage_account = "team08" # The name of your Storage account created in https://portal.azure.com
secret_scope = "team08-scope" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "team08-key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# COMMAND ----------

spark

# COMMAND ----------

# Inspect the Mount's Final Project folder 
display(dbutils.fs.ls("/mnt/mids-w261/datasets_final_project"))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Common Functions

# COMMAND ----------

# Load the 2015 Q1 for Weather
def read_weather_data(cut_off_date):
    filter_date = cut_off_date + "T00:00:00.000"
    df_weather = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*").filter(col('DATE') < filter_date)
    return df_weather

# COMMAND ----------

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

def null_values_eda(df):
  total_count = df.count()
  for column in df.columns:
    count = df.filter(df[column].isNull()).count()
    print(column, ":", count, ":", total_count, ":", count * 100 / total_count)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 1. Load Dataset

# COMMAND ----------

df_weather_3m = read_weather_data("2015-01-02")
#df_weather_6m = read_weather_data("2015-07-01")
#display(df_weather_3m)

# COMMAND ----------

drop_additional_weather_columns = {"AW1","GA1","GA2","GA3","GA4","GE1","GF1","KA2","MA1","OD1","OD2","REM","EQD","AW2","AX4","GD1","AW5","GN1","AJ1","AW3","MK1","KA4","GG3","AN1","RH1","AU5","HL1","OB1","AT8","AW7","AZ1","CH1","RH3","GK1","IB1","AX1","CT1","AK1","CN2","OE1","MW5","AO1","KA3","AA3","CR1","CF2","KB2","GM1","AT5","MW6","MG1","AH6","AU2","GD2","AW4","MF1","AH2","AH3","OE3","AT6","AL2","AL3","AX5","IB2","AI3","CV3","WA1","GH1","KF1","CU2","CT3","SA1","AU1","KD2","AI5","GO1","GD3","CG3","AI1","AL1","AW6","MW4","AX6","CV1","ME1","KC2","CN1","UA1","GD5","UG2","AT3","AT4","GJ1","MV1","GA5","CT2","CG2","ED1","AE1","CO1","KE1","KB1","AI4","MW3","KG2","AA2","AX2","RH2","OE2","CU3","MH1","AM1","AU4","GA6","KG1","AU3","AT7","KD1","GL1","IA1","GG2","OD3","UG1","CB1","AI6","CI1","CV2","AZ2","AD1","AH1","WD1","AA4","KC1","IA2","CF3","AI2","AT1","GD4","AX3","AH4","KB3","CU1","CN4","AT2","CG1","CF1","GG1","MV2","CW1","GG4","AB1","AH5","CN3", "AY2", "KA1"}

df_weather_3m = drop_columns(df_weather_3m, drop_additional_weather_columns)
#df_weather_6m = drop_columns(df_weather_6m, drop_additional_weather_columns)
#display(df_weather_3m)

# COMMAND ----------

df_weather_3m = split_columns(df_weather_3m)

# COMMAND ----------

drop_cols={"SLP","DEW","TMP","VIS","CIG","WND","MD1","MW1","MW2","OC1","AA1","AY1","LATITUDE","LONGITUDE","SOURCE","REPORT_TYPE","QUALITY_CONTROL","NAME","CALL_SIGN","AIR_TEMP_QUALITY_CODE","DEW_POINT_QUALITY_CODE","SEA_LEVEL_PRESSURE_QUALITY_CODE","WIND_DIR_QUALITY_CODE","WIND_SPEED_QUALITY_CODE","VIS_DISTANCE_QUALITY_CODE","CEILING_QUALITY_CODE","CEILING_DETERM_CODE","CAVOK","VIS_DISTANCE_QUALITY_CODE","VIS_QUALITY_VARIABILITY_CODE","VIS_VARIABILTY_CODE"}

df_weather_3m = drop_columns(df_weather_3m, drop_cols)
# df_weather_6m_augmented = drop_columns(df_weather_3m, drop_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Weather DataFrame with Raw Features

# COMMAND ----------

display(df_weather_3m)

# COMMAND ----------

df_weather_3m.printSchema()

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Step 2. Exploratory Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Summary Statistics

# COMMAND ----------

summary = df_weather_6m_augmented.summary()

# COMMAND ----------

display(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Categorical Features Distribution

# COMMAND ----------

PRESENT_ATMOSPHERIC_CONDITION_CODE = df_weather_6m_augmented.select(col("PRESENT_ATMOSPHERIC_CONDITION_CODE").cast("string")).groupBy("PRESENT_ATMOSPHERIC_CONDITION_CODE").count()

PAST_ATMOSPHERIC_CONDITION_CODE = df_weather_6m_augmented.select(col("PAST_ATMOSPHERIC_CONDITION_CODE").cast("string")).groupBy("PAST_ATMOSPHERIC_CONDITION_CODE").count()

PAST_ATMOSPHERIC_CONDITION_CODE_DURATION = df_weather_6m_augmented.select(col("PAST_ATMOSPHERIC_CONDITION_CODE_DURATION").cast("string")).groupBy("PAST_ATMOSPHERIC_CONDITION_CODE_DURATION").count()

ATMOS_PRESSURE_TENDENCY_3H = df_weather_6m_augmented.select(col("ATMOS_PRESSURE_TENDENCY_3H").cast("string")).groupBy("ATMOS_PRESSURE_TENDENCY_3H").count()

PRECIPITATION_PERIOD_QTY = df_weather_6m_augmented.select(col("PRECIPITATION_PERIOD_QTY").cast("string")).groupBy("PRECIPITATION_PERIOD_QTY").count()

PRECIPITATION_DEPTH = df_weather_6m_augmented.select(col("PRECIPITATION_DEPTH").cast("string")).groupBy("PRECIPITATION_DEPTH").count()

# COMMAND ----------

display(PRESENT_ATMOSPHERIC_CONDITION_CODE)

# COMMAND ----------

display(PAST_ATMOSPHERIC_CONDITION_CODE)

# COMMAND ----------

display(PAST_ATMOSPHERIC_CONDITION_CODE_DURATION)

# COMMAND ----------

display(ATMOS_PRESSURE_TENDENCY_3H)

# COMMAND ----------

display(PRECIPITATION_PERIOD_QTY)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Nulls and Missing values

# COMMAND ----------

null_values_eda(df_weather_6m_augmented)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Distribution of Missing Values (9,99,999,9999,99999,999999)

# COMMAND ----------

Group_9 = { "WIND_TYPE" , "ATMOS_PRESSURE_TENDENCY_3H"}
Group_99 = { "PAST_ATMOSPHERIC_CONDITION_CODE_DURATION", "PRECIPITATION_PERIOD_QTY" }	
Group_999 ={ "WIND_DIR", "ATMOS_PRESSURE_3H" }	
Group_9999 = { "AIR_TEMP", "DEW_POINT_TEMPERATURE", "ELEVATION", "WIND_SPEED_RATE" , "WIND_GUST_SPEED_RATE", "PRECIPITATION_DEPTH"}	
Group_99999	= { "CEILING_HEIGHT", "SEA_LEVEL_PRESSURE" }
Group_999999 = { "VIS_DISTANCE"}

for column in Group_9:
    count = df_weather_6m_augmented.filter(df_weather_6m_augmented[column]=="9").count()
    print(column, ":", count)

# COMMAND ----------

for column in Group_99:
    count = df_weather_6m_augmented.filter(df_weather_6m_augmented[column]=="99").count()
    print(column, ":", count)

# COMMAND ----------

for column in Group_999:
    count = df_weather_6m_augmented.filter(df_weather_6m_augmented[column]=="999").count()
    print(column, ":", count)

# COMMAND ----------

for column in Group_9999:
    count = df_weather_6m_augmented.filter(df_weather_6m_augmented[column]=="9999").count()
    print(column, ":", count)

# COMMAND ----------

for column in Group_99999:
    count = df_weather_6m_augmented.filter(df_weather_6m_augmented[column]=="99999").count()
    print(column, ":", count)

# COMMAND ----------

for column in Group_999999:
    count = df_weather_6m_augmented.filter(df_weather_6m_augmented[column]=="999999").count()
    print(column, ":", count)

# COMMAND ----------

for column in {"DEW_POINT_TEMPERATURE", "AIR_TEMP"}:
    count = df_weather_6m_augmented.filter(df_weather_6m_augmented[column]=="+9999").count()
    print(column, ":", count)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 3. Data Cleaning

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Drop Records with missing STATION

# COMMAND ----------

def drop_missing_station_records(df):
  return df.na.drop(subset=["STATION"])

df_weather_3m = drop_missing_station_records(df_weather_3m)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Convert Data Types

# COMMAND ----------

# feature_mcode_mapping = {"WIND_DIR":999, "WIND_SPEED_RATE":9999, "VIS_DISTANCE":999999,"CEILING_HEIGHT":99999, "AIR_TEMP":9999, "DEW_POINT_TEMPERATURE":9999, "SEA_LEVEL_PRESSURE":99999, "ATMOS_PRESSURE_3H":999 }

df_weather_3m = df_weather_3m \
                              .withColumn("WIND_DIR",df_weather_3m.WIND_DIR.cast(IntegerType())) \
                              .withColumn("WIND_SPEED_RATE",df_weather_3m.WIND_SPEED_RATE.cast(IntegerType())) \
                              .withColumn("CEILING_HEIGHT",df_weather_3m.CEILING_HEIGHT.cast(IntegerType())) \
                              .withColumn("VIS_DISTANCE",df_weather_3m.VIS_DISTANCE.cast(IntegerType())) \
                              .withColumn("AIR_TEMP",df_weather_3m.AIR_TEMP.cast(IntegerType())) \
                              .withColumn("DEW_POINT_TEMPERATURE",df_weather_3m.DEW_POINT_TEMPERATURE.cast(IntegerType())) \
                              .withColumn("SEA_LEVEL_PRESSURE",df_weather_3m.SEA_LEVEL_PRESSURE.cast(IntegerType())) \
                              .withColumn("ATMOS_PRESSURE_3H",df_weather_3m.ATMOS_PRESSURE_3H.cast(IntegerType()))
                            

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Average Imputation

# COMMAND ----------

feature_mcode_mapping = {"WIND_DIR":999, "WIND_SPEED_RATE":9999, "VIS_DISTANCE":999999,"CEILING_HEIGHT":99999, "AIR_TEMP":9999, "DEW_POINT_TEMPERATURE":9999, "SEA_LEVEL_PRESSURE":99999, "ATMOS_PRESSURE_3H":999 }

def replace(column, value):
    return when(column != value, column).otherwise(lit(None))

for feature, missing_code in feature_mcode_mapping.items():
    df_weather_3m = df_weather_3m.withColumn(feature, replace(col(feature), missing_code))

# COMMAND ----------

def apply_avg_imputation(df, col_name, missing_code):
  window = Window.partitionBy("STATION").orderBy("DATE").rowsBetween(-720, 0)
  condition = when( (col(col_name).isNull()) | (col(col_name) == missing_code), floor(mean(col(col_name)).over(window))).otherwise(col(col_name))
  return df.withColumn(col_name+"_IMP", condition)          
 

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql.functions import avg,mean

feature_mcode_mapping = {"WIND_DIR":999, "WIND_SPEED_RATE":9999, "VIS_DISTANCE":999999,"CEILING_HEIGHT":99999, "AIR_TEMP":9999, "DEW_POINT_TEMPERATURE":9999, "SEA_LEVEL_PRESSURE":99999, "ATMOS_PRESSURE_3H":999 }

for column, missing_code in feature_mcode_mapping.items():
  df_weather_3m=apply_avg_imputation(df_weather_3m, column, missing_code)

# COMMAND ----------

def apply_avg_imputation_across_feature(df, feature_name):
  window = Window.partitionBy()
  condition = when(col(feature_name).isNull(), floor(mean(col(feature_name)).over(window))).otherwise(col(feature_name))
  return df.withColumn(feature_name, condition)      

for feature, missing_code in feature_mcode_mapping.items():
  df_weather_3m=apply_avg_imputation_across_feature(df_weather_3m, feature+"_IMP")

# COMMAND ----------

display(df_weather_3m)

# COMMAND ----------

from pyspark.sql.functions import col,when
df_weather_3m = df_weather_3m \
                                .withColumn("PRECIPITATION_PERIOD_QTY", when(col("PRECIPITATION_PERIOD_QTY")=="" ,"99").otherwise(col("PRECIPITATION_PERIOD_QTY"))) \
                                .withColumn("WIND_GUST_SPEED_RATE", when(col("WIND_GUST_SPEED_RATE")=="" ,"0").otherwise(col("WIND_GUST_SPEED_RATE"))) \
                                .withColumn("WIND_TYPE", when(col("WIND_SPEED_RATE")=="0000" ,"C").otherwise(col("WIND_TYPE"))) \
                                .withColumn("PRESENT_ATMOSPHERIC_CONDITION_CODE", when(col("PRESENT_ATMOSPHERIC_CONDITION_CODE")=="" ,"NA").otherwise(col("PRESENT_ATMOSPHERIC_CONDITION_CODE"))) \
                                .withColumn("PAST_ATMOSPHERIC_CONDITION_CODE", when(col("PAST_ATMOSPHERIC_CONDITION_CODE")=="" ,"NA").otherwise(col("PAST_ATMOSPHERIC_CONDITION_CODE"))) \
                                .withColumn("ATMOS_PRESSURE_TENDENCY_3H", when(col("ATMOS_PRESSURE_TENDENCY_3H")=="" ,"NA").otherwise(col("ATMOS_PRESSURE_TENDENCY_3H"))) \
                                .withColumn("WIND_TYPE", when(col("WIND_TYPE")=="" ,"NA").otherwise(col("WIND_TYPE")))

# COMMAND ----------

df_weather_3m = df_weather_3m \
                              .withColumn("WIND_GUST_SPEED_RATE",df_weather_3m.WIND_GUST_SPEED_RATE.cast(IntegerType())) \
                              .withColumn("PRECIPITATION_DEPTH",df_weather_3m.PRECIPITATION_DEPTH.cast(IntegerType())) \
                              .withColumn("PRECIPITATION_PERIOD_QTY",df_weather_3m.PRECIPITATION_PERIOD_QTY.cast(IntegerType())) \
                              .withColumn("PAST_ATMOSPHERIC_CONDITION_CODE_DURATION",df_weather_3m.PAST_ATMOSPHERIC_CONDITION_CODE_DURATION.cast(IntegerType()))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### One Hot Encoding

# COMMAND ----------

from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder

indexer = StringIndexer(inputCols=["PRESENT_ATMOSPHERIC_CONDITION_CODE","PAST_ATMOSPHERIC_CONDITION_CODE","ATMOS_PRESSURE_TENDENCY_3H","WIND_TYPE"], outputCols=["PRESENT_ACC_INDEX","PAST_ACC_INDEX","ATMOS_PRESSURE_TENDENCY_3H_INDEX","WIND_TYPE_INDEX"])
df_weather_3m = indexer.fit(df_weather_3m).transform(df_weather_3m)

ohe = OneHotEncoder(inputCols=["PRESENT_ACC_INDEX","PAST_ACC_INDEX","ATMOS_PRESSURE_TENDENCY_3H_INDEX","WIND_TYPE_INDEX"], outputCols=["PRESENT_ACC_OHE","PAST_ACC_OHE","ATMOS_PRESSURE_TENDENCY_3H_OHE","WIND_TYPE_OHE"])
df_weather_3m=ohe.fit(df_weather_3m).transform(df_weather_3m)

cols = {"PRESENT_ATMOSPHERIC_CONDITION_CODE","PAST_ATMOSPHERIC_CONDITION_CODE","ATMOS_PRESSURE_TENDENCY_3H","WIND_TYPE","PRESENT_ACC_INDEX","PAST_ACC_INDEX","ATMOS_PRESSURE_TENDENCY_3H_INDEX","WIND_TYPE_INDEX"}
df_weather_3m=drop_columns(df_weather_3m,cols)

# COMMAND ----------

cols = { "WIND_DIR", "WIND_SPEED_RATE", "VIS_DISTANCE","CEILING_HEIGHT","AIR_TEMP", "DEW_POINT_TEMPERATURE", "SEA_LEVEL_PRESSURE", "ATMOS_PRESSURE_3H"}
df_weather_3m=drop_columns(df_weather_3m,cols)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 4. Review Final Dataframe

# COMMAND ----------

display(df_weather_3m)

# COMMAND ----------

df_weather_6m_augmented.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 5. Write data to blob storage in Parquet format

# COMMAND ----------

df_weather_3m.write.parquet(f"{blob_url}/df_weather_3m_eda_final")

# COMMAND ----------

### NOTE: Augmented dataframe below includes additional weather attributes such as Snow, Precipitation, Thunder, Tornadoes, etc. UNCOMMENT AFTER PHASE 2 and comment out the previous cell

#df_weather_3m_augmented.write.parquet(f"{blob_url}/df_weather_3m")
#df_weather_6m_augmented.write.parquet(f"{blob_url}/df_weather_6m")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Appendix A : List of Features from Weather Data Set
# MAGIC 
# MAGIC | Variable | DataType | Data Source | Missing % | Preprocessing |
# MAGIC |---|---|---|---|---|
# MAGIC |STATION|Categorical/Nominal|Available in Weather Dataset|0.61%|Preserve data type as String of Integers|
# MAGIC |DATE|TimeStamp|Available in Weather Dataset|0%|||
# MAGIC |ELEVATION|Numeric|Available in Weather Dataset|0%|String2Int, Discretize|
# MAGIC |WIND_DIR|Numeric|Available in Weather Dataset|0%|*Convert String to Numeric, contains Erroneous Data, No Preprocessing* |
# MAGIC |WIND_TYPE|Nominal|Available in Weather Dataset|16.23%|Though 16% of data is missing, convert to WIND_TYPE CALM (C) if WIND_SPEED_RATE is 0.|
# MAGIC |WIND_SPEED_RATE|Numeric|Available in Weather Dataset| 0% ||
# MAGIC |CELING_HEIGHT|Numeric|Available in Weather Dataset|48.0349%||
# MAGIC |VIS_DISTANCE|Numeric|Available in Weather Dataset|32.35%||
# MAGIC |AIR_TEMP|Numeric|Available in Weather Dataset|1.5%||
# MAGIC |DEW_POINT_TEMP|Numeric|Available in Weather Dataset|15.42% ||
# MAGIC |SEA_LEVEL_PRESSURE|Numeric|Available in Weather Dataset|65.14%||
# MAGIC 
# MAGIC ### Additional Attributes
# MAGIC 
# MAGIC **MW1**
# MAGIC PRESENT-WEATHER-OBSERVATION **Detailed Codes**
# MAGIC 
# MAGIC | Variable | DataType | Data Source | Missing % | Preprocessing |
# MAGIC |---|---|---|---|---|
# MAGIC |PRESENT_ATMOSPHERIC_CONDITION_CODE||MW1||
# MAGIC 
# MAGIC **AY1, AY2**
# MAGIC Past Weather Observations - contains indicators for fog, drizzle, rain, showers. 
# MAGIC 
# MAGIC | Variable | DataType | Data Source | Missing % | Preprocessing |
# MAGIC |---|---|---|---|---|
# MAGIC |PAST_ATMOSPHERIC_CONDITION_CODE|Ordinal|AY1||
# MAGIC |PAST_ATMOSPHERIC_CONDITION_CODE_PERIOD|Numeric|AY1||
# MAGIC 
# MAGIC **MD1**
# MAGIC ATMOSPHERIC-PRESSURE-CHANGE
# MAGIC Captures the tendancy of atmospheric pressure - is it increasing, decreasing, stable in comparison to last 3 hours and 24 hours.
# MAGIC 
# MAGIC | Variable | DataType | Data Source | Missing % | Preprocessing |
# MAGIC |---|---|---|---|---|
# MAGIC |MD1_ATM_PRESSURE_TENDENCY_CODE||||
# MAGIC |MD1_ATM_PRESSURE_TENDENCY_3H||||
# MAGIC |MD1_ATM_PRESSURE_TENDENCY_24H||||
# MAGIC 
# MAGIC **OC1**
# MAGIC WIND-GUST-OBSERVATION 
# MAGIC 
# MAGIC | Variable | DataType | Data Source | Missing % | Preprocessing |
# MAGIC |---|---|---|---|---|
# MAGIC |WIND_GUST_SPEED_RATE||||
# MAGIC 
# MAGIC **AA1, AA2**
# MAGIC PRECIPITATION Depth (amount of prec measured)
# MAGIC 
# MAGIC | Variable | DataType | Data Source | Missing % | Preprocessing |
# MAGIC |---|---|---|---|---|
# MAGIC |AA1_PERIOD_QTY||AA1||
# MAGIC |AA1_DEPTH_DIM||AA1||
# MAGIC |AA1_CONDITION_CODE||AA1||
# MAGIC |AA2_PERIOD_QTY||AA1||
# MAGIC |AA2_DEPTH_DIM||AA1||
# MAGIC |AA2_CONDITION_CODE||AA1||
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC **KA1** EXTREME-AIR-TEMPERATURE
# MAGIC 
# MAGIC | Variable | DataType | Data Source | Missing % | Preprocessing |
# MAGIC |---|---|---|---|---|
# MAGIC |X_AIR_TEMP_DURATION||KA1||
# MAGIC |X_AIR_TEMP||KA1||
# MAGIC 
# MAGIC **GA1 to GA3** - representative of the cloud cover in the sky, the type of clouds in the sky
# MAGIC The the three GA sets 
# MAGIC 
# MAGIC | Variable | DataType | Data Source | Missing % | Preprocessing |
# MAGIC |---|---|---|---|---|
# MAGIC |GA1_COVERAGE_CODE|Ordinal|GA1|||
# MAGIC |GA1_BASE_HEIGHT_DIM|Ordinal|GA1|||
# MAGIC |GA1_CLOUD_TYPE_CODE|Ordinal|GA1|||
# MAGIC |GA2_COVERAGE_CODE|Ordinal|GA2|||
# MAGIC |GA2_BASE_HEIGHT_DIM|Ordinal|GA2|||
# MAGIC |GA2_CLOUD_TYPE_CODE|Ordinal|GA2|||
# MAGIC |GA3_COVERAGE_CODE|Ordinal|GA3|||
# MAGIC |GA3_BASE_HEIGHT_DIM|Ordinal|GA3|||
# MAGIC |GA3_CLOUD_TYPE_CODE|Ordinal|GA3|||
# MAGIC 
# MAGIC **OD1, OD2** SUPPLEMENTARY-WIND-OBSERVATION 
# MAGIC Contains Maximum and Mean Wind Speeds for different wind types.
# MAGIC 
# MAGIC | Variable | DataType | Data Source | Missing % | Preprocessing |
# MAGIC |---|---|---|---|---|
# MAGIC |OD1_WIND_TYPE_CODE||OD1||
# MAGIC |OD1_WIND_PERIOD_QTY||OD1||
# MAGIC |OD1_WIND_DIR_QTY||OD1||
# MAGIC |OD1_WIND_SPEED_RATE||OD1||
# MAGIC |OD2_WIND_TYPE_CODE||OD2||
# MAGIC |OD2_WIND_PERIOD_QTY||OD2||
# MAGIC |OD2_WIND_DIR_QTY||OD2||
# MAGIC |OD2_WIND_SPEED_RATE||OD2||

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Appendix B : Missing Values Table
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

# COMMAND ----------

from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
import seaborn as sns


df_airlines_weather_full = spark.read.parquet(f"{blob_url}/df_airlines_weather_holidays")
df_airlines_weather_full.cache()
df_airlines_weather = df_airlines_weather_full.where(df_airlines_weather_full.FL_DATE_UTC < '2019-01-01')
df_airlines_weather.cache()



# COMMAND ----------

#df_weather = spark.read.parquet(f"{blob_url}/df_weather")

# convert to vector column first
vector_col = "corr_features"
inputColumns = ["WIND_DIR_IMP", "WIND_SPEED_RATE_IMP", "CEILING_HEIGHT_IMP", "VIS_DISTANCE_IMP", "AIR_TEMP_IMP", "DEW_POINT_TEMPERATURE_IMP", "SEA_LEVEL_PRESSURE_IMP", "DEP_DEL15"]
assembler = VectorAssembler(inputCols=inputColumns, outputCol=vector_col, handleInvalid='skip')
df_vector = assembler.transform(df_airlines_weather).select(vector_col)

# COMMAND ----------

# get correlation matrix
matrix = Correlation.corr(df_vector, vector_col).collect()[0][0]

corrmatrix = matrix.toArray().tolist()
cmap = sns.light_palette("#4e909e", as_cmap=True)


# COMMAND ----------

import matplotlib.pyplot as plt
plt.title("Correlation Matrix", fontsize =20)
sns.heatmap(corrmatrix, 
        xticklabels=inputColumns,
        yticklabels=inputColumns, cmap=cmap, annot=True, vmin=-1, vmax=1)

# COMMAND ----------

display(df_airlines_weather)