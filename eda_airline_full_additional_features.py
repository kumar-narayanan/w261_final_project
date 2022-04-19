# Databricks notebook source
# MAGIC %md
# MAGIC # EDA for Full Airline Data

# COMMAND ----------

from functools import reduce
from pyspark.sql import DataFrame
from pyspark.sql.functions import col,split,isnan, when, count, concat, lit, desc, floor, hour, sum, lag, last, mean
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import SQLContext
from pyspark.sql import DataFrameNaFunctions
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Binarizer
from pyspark.ml.feature import OneHotEncoder, VectorAssembler, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

# DBTITLE 1,Init storage account
from pyspark.sql.functions import col, max

blob_container = "team08-container" # The name of your container created in https://portal.azure.com
storage_account = "team08" # The name of your Storage account created in https://portal.azure.com
secret_scope = "team08-scope" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "team08-key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# COMMAND ----------

# Inspect the Mount's Final Project folder 
display(dbutils.fs.ls("/mnt/mids-w261/datasets_final_project/parquet_airlines_data/"))

# COMMAND ----------

# MAGIC %md ## Load Airlines Data

# COMMAND ----------

# MAGIC %md Read from parquet and load the full airlines dataset.

# COMMAND ----------

# DBTITLE 1,Load the full airlines data
df_airlines = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data/20*")

# COMMAND ----------

display(df_airlines)

# COMMAND ----------

df_airlines.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## List of Features in Airlines Data Set
# MAGIC The features in the flights dataset can be categorized into different groups as given below:
# MAGIC 
# MAGIC |Group| Features|
# MAGIC |-|-|
# MAGIC |Temporal Features| Year, Quarter, Month, Day_Of_Month, Day_Of_Week, FL_Date|
# MAGIC |Operator/Carrier Features| Op_Unique_Carrier, Op_Carrier_Airline_ID, Op_Carrier, Tail_Num, Op_Carrier_Fl_Num|
# MAGIC |Origin Airport Features| Origin_Airport_Id, Origin_Airport_Seq_Id, Origin_City_Market_Id, Origin, Origin_City_Name, Origin_State_Abr, Origin_State_Fips, Origin_State_Nm, Origin_Wac|
# MAGIC |Destination Airport Features|Dest_Airport_Id, Dest_Airport_Seq_Id, Dest_City_Market_Id, Dest, Dest_City_Name, Dest_State_Abr, Dest_State_Fips, Dest_State_Name, Dest_Wac|
# MAGIC |Departure Performance| CRS_Dep_Time, Dep_time, Dep_Delay, Dep_Delay_New, Dep_Del15, Dep_Delay_Group, Dep_Time_Blk, Taxi_Out, Wheels_Off|
# MAGIC |Arrival Performance| CRS_Arr_Time, Arr_Time, Arr_Delay, Arr_Delay_New, Arr_Del15, Arr_Delay_Group, Arr_Time_Blk, Wheels_On, Taxi_In|
# MAGIC |Cancellation and Diversions| Cancelled, Cancellation Code, Diverted|
# MAGIC |Flight Summaries| CRS_ELAPSED_TIME, ACTUAL_ELAPSED_TIME, AIR_TIME, FLIGHTS, DISTANCE, DISTANCE_GROUP|
# MAGIC |Cause of Delay| CARRIER_DELAY, WEATHER_DELAY, NAS_DELAY, SECURITY_DELAY, LATE_AIRCRAFT_DELAY|
# MAGIC |Gate Return Information at Origin Airport| FIRST_DEP_TIME, TOTAL_ADD_GTIME, LONGEST_ADD_GTIME|
# MAGIC |Diverted Airport Information| DIV_AIRPORT_LANDINGS, DIV_REACHED_DEST, DIV_ACTUAL_ELAPSED_TIME, DIV_ARR_DELAY, DIV_DISTANCE, DIV1_AIRPORT, DIV1_AIRPORT_ID, DIV1_AIRPORT_SEQ_ID, DIV1_WHEELS_ON, DIV1_TOTAL_GTIME, DIV1_LONGEST_GTIME, DIV1_WHEELS_OFF, DIV1_TAIL_NUM, DIV2_AIRPORT, DIV2_AIRPORT_ID, DIV2_AIRPORT_SEQ_ID, DIV2_WHEELS_ON, DIV2_TOTAL_GTIME, DIV2_LONGEST_GTIME, DIV2_WHEELS_OFF, DIV2_TAIL_NUM, DIV3_AIRPORT, DIV3_AIRPORT_ID, DIV3_AIRPORT_SEQ_ID, DIV3_WHEELS_ON, DIV3_TOTAL_GTIME, DIV3_LONGEST_GTIME, DIV3_WHEELS_OFF, DIV3_TAIL_NUM, DIV4_AIRPORT, DIV4_AIRPORT_ID, DIV4_AIRPORT_SEQ_ID, DIV4_WHEELS_ON, DIV4_TOTAL_GTIME, DIV4_LONGEST_GTIME, DIV4_WHEELS_OFF, DIV4_TAIL_NUM, DIV5_AIRPORT,  DIV5_AIRPORT_ID, DIV5_AIRPORT_SEQ_ID, DIV5_WHEELS_ON, DIV5_TOTAL_GTIME, DIV5_LONGEST_GTIME, DIV5_WHEELS_OFF, DIV5_TAIL_NUM

# COMMAND ----------

# MAGIC %md ## Summary Statistics

# COMMAND ----------

# DBTITLE 1,Statistics of the Airlines Table
summary = df_airlines.summary()

# COMMAND ----------

display(summary)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC From the summary table above, we see that that from the operator/carrier features, TAIL_NUM is missing values, from the departure performance features, DEP_TIME, DEP_DELAY, DEP_DELAY_NEW, DEP_DEL15, DEP_DELAY_GROUP, TAXI_OUT, WHEELS_OFF do not have values for all entries. Among the arrival performance features, ARR_TIME, ARR_DELAY, ARR_DELAY_NEW, ARR_DEL15, ARR_DELAY_GROUP, TAXI_IN, WHEELS_ON do not have values for all entries. In the Flight Summaries group, CRS_ELAPSED_TIME, ACTUAL_ELAPSED_TIME, AIR_TIME are missing values for some records. Understandably, the cause of delay group features, CARRIER_DELAY, WEATHER_DELAY, NAS_DELAY, SECURITY_DELAY, LATE_AIRCRAFT_DELAY have values for only a few records. In the Gate Return Information feature group, FIRST_DEP_TIME, TOTAL_ADD_GTIME, LONGEST_ADD_GTIME is available only for a few records. In the Diverted Airport Information feature group, majority of these features have no values indicating that a very small fraction of flights were diverted.

# COMMAND ----------

df_airlines.count()

# COMMAND ----------

# MAGIC %md ## Correlation Matrix

# COMMAND ----------

from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

# convert to vector column first
vector_col = "corr_features"
inputColumns = ["YEAR", "QUARTER", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK"]
assembler = VectorAssembler(inputCols=inputColumns, outputCol=vector_col)
df_vector = assembler.transform(df_airlines).select(vector_col)

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

# MAGIC %md ## Missing Values

# COMMAND ----------

# DBTITLE 1,Columns with missing values
def null_values_eda(df):
  total_count = df.count()
  print(f"|Column| count| percent|")
  print(f"|-|-|-|")
  for column in df.columns:
    count = df.where(col(column).isNull()).count()
    if count > 0:
      print(f"|{column}|{count}|{count*100/float(total_count):.3f}%|")

# COMMAND ----------

null_values_eda(df_airlines)

# COMMAND ----------

# MAGIC %md ## Categorical Variables

# COMMAND ----------

pd.set_option('display.max_rows', None)
categorical_variables = ['Year', 'Quarter', 'Month', 'Day_Of_Month', 'Day_Of_Week', 'Op_Unique_Carrier', 'Origin', 'Dest', "Origin-Dest", "Cancellation_Code"]

for column in categorical_variables:
  freq_table = df_airlines.withColumn("Origin-Dest", concat(col("Origin"), lit("-"),col("Dest"))).select(col(column).cast("string")).groupBy(column).count().sort(desc("count")).toPandas()
  print(f"Frequency Distribution for column {column}")
  print(freq_table)
  print("\n")

# COMMAND ----------

# MAGIC %md ### Year
# MAGIC 
# MAGIC We see that there has been a slight increase in flights in 2018 and 2019

# COMMAND ----------

display(df_airlines.select("Year", "Dep_Del15").groupBy("Year", "Dep_Del15").count().sort("Year"))

# COMMAND ----------

# MAGIC %md ### Quarter
# MAGIC 
# MAGIC We see that there is a slight increase in flights during the 2nd and 3rd quarters.

# COMMAND ----------

display(df_airlines.select("Quarter", "Dep_Del15").groupBy("Quarter", "Dep_Del15").count().sort("Quarter"))

# COMMAND ----------

# MAGIC %md ### Month
# MAGIC 
# MAGIC We see that the there are slightly more flights in July and August.

# COMMAND ----------

display(df_airlines.select("Month", "Dep_Del15").groupBy("Month", "Dep_Del15").count().sort("Month"))

# COMMAND ----------

# MAGIC %md ### Day of Month
# MAGIC 
# MAGIC We see that the airlines data seems to be spread equally across all days of the month. Though there seems to be fewer flights on the 31st, it is probably due to the fact that only a few months have 31 days.

# COMMAND ----------

display(df_airlines.select("Day_Of_Month", "Dep_Del15").groupBy("Day_Of_Month", "Dep_Del15").count().sort("Day_Of_Month"))

# COMMAND ----------

# MAGIC %md ### Day of Week
# MAGIC 
# MAGIC We see that the airlines data seems to be spread equally across all days of the week

# COMMAND ----------

display(df_airlines.select("Day_Of_Week", "Dep_Del15").groupBy("Day_Of_Week", "Dep_Del15").count().sort("Day_Of_Week"))

# COMMAND ----------

# MAGIC %md ### Hour of Day
# MAGIC 
# MAGIC We see that the flights tend to be more delayed during the evenings.

# COMMAND ----------

display(df_airlines.withColumn('Hour',  floor(col('CRS_Dep_Time')/100)).select("Hour", "Dep_Del15").groupBy("Dep_Del15", "Hour").count().sort("hour", "Dep_Del15"))

# COMMAND ----------

# MAGIC %md ### Op_Carrier
# MAGIC 
# MAGIC There are a few operators such as AA (American Airlines), DL (Delta Airlines), WN (SouthWest Airlines) who have more flights than other operators

# COMMAND ----------

display(df_airlines.select("Op_Carrier").groupBy("Op_Carrier").count().sort("Op_Carrier"))

# COMMAND ----------

# MAGIC %md ### Op_Carrier_Fl_Num

# COMMAND ----------

display(df_airlines.select("Op_Carrier_Fl_Num").groupBy("Op_Carrier_Fl_Num").count().sort(desc("count")))

# COMMAND ----------

# MAGIC %md ###Busy Airports

# COMMAND ----------

df_origin_airports = df_airlines.select(col("Origin").alias("Airport")).groupBy("Airport").count().sort(desc("count"))
df_dest_airports = df_airlines.select(col("Dest").alias("Airport")).groupBy("Airport").count().sort(desc("count"))

# COMMAND ----------

df_busy_airports = reduce(DataFrame.unionAll, [df_origin_airports, df_dest_airports])
display(df_busy_airports.groupby("airport").agg(sum("count").alias("total")).sort(desc("total")).limit(10))

# COMMAND ----------

# MAGIC %md ### Origin-Destination
# MAGIC 
# MAGIC We see that there are some origin-destination airport pairs where the air traffic or number of flights is highest such as SFO-LAX, LAX-JFK

# COMMAND ----------

display(
  df_airlines.withColumn("Origin-Dest", concat(col("Origin"), lit("-"), col("Dest")))
  .select("Origin-Dest", "Dep_Del15")
  .groupBy("Origin-Dest", "Dep_Del15")
  .count()
  .sort(desc("count")))

# COMMAND ----------

# MAGIC %md ### DEP_DEL15
# MAGIC 
# MAGIC Around 477,296 records in the airline data do not have any departure delay status. Also majority of flights do not have a delay.

# COMMAND ----------

display(df_airlines.select("DEP_DEL15").groupBy("DEP_DEL15").count())

# COMMAND ----------

# MAGIC %md ### CANCELLATION_CODE
# MAGIC 
# MAGIC Majority of the flights are not cancelled. However, we also notice that CANCELLATION_CODE **"B"** has more entries that the other codes "A", "C" and "D".

# COMMAND ----------

display(df_airlines.select("CANCELLATION_CODE").groupBy("CANCELLATION_CODE").count())

# COMMAND ----------

# MAGIC %md ### DEP_DEL15 and CANCELLATION_CODE
# MAGIC 
# MAGIC There are 9488 flights that do not have a departure delay status nor a cancellation code.

# COMMAND ----------

display(df_airlines.select("DEP_DEL15", "CANCELLATION_CODE").groupBy("DEP_DEL15", "CANCELLATION_CODE").count().sort("DEP_DEL15", "CANCELLATION_CODE"))

# COMMAND ----------

# MAGIC %md ### DIV_AIRPORT_LANDINGS
# MAGIC 
# MAGIC While a majority of the flights have 0 airport diversions, among the diverted airport landings, the most number of diverted airport landings is 1. 

# COMMAND ----------

display(df_airlines.select("DIV_AIRPORT_LANDINGS").groupBy("DIV_AIRPORT_LANDINGS").count())

# COMMAND ----------

# MAGIC %md ### ARR_TIME

# COMMAND ----------

display(df_airlines.where(col("ARR_TIME").isNull()))

# COMMAND ----------

display(df_airlines.where((col("ARR_TIME").isNull()) & (col('CANCELLED') != 1) & (col('DIVERTED') != 1)))

# COMMAND ----------

display(df_airlines.where( (col("ARR_TIME") < 0) | (col("ARR_TIME") > 2359)))

# COMMAND ----------

df_airlines.where(col("ARR_TIME") > 2359).count()

# COMMAND ----------

df_airlines.where(col("ARR_TIME") == 2400).count()

# COMMAND ----------

# MAGIC %md ## Numerical Variables

# COMMAND ----------

# MAGIC %md ### Dep_Delay

# COMMAND ----------

display(df_airlines.select("Dep_Delay").where(col("Dep_Delay") > 0).groupBy("Dep_Delay").count().sort(desc("count")))

# COMMAND ----------

# MAGIC %md ### CARRIER_DELAY
# MAGIC 
# MAGIC We see most carrier delays are short duration.

# COMMAND ----------

display(df_airlines.select("CARRIER_DELAY").where(col("CARRIER_DELAY") > 0))

# COMMAND ----------

# MAGIC %md ### WEATHER_DELAY

# COMMAND ----------

display(df_airlines.select("WEATHER_DELAY").where(col("WEATHER_DELAY") > 0))

# COMMAND ----------

# MAGIC %md ### NAS_DELAY

# COMMAND ----------

display(df_airlines.select("NAS_DELAY").where(col("NAS_DELAY") > 0))

# COMMAND ----------

# MAGIC %md ### SECURITY_DELAY

# COMMAND ----------

display(df_airlines.select("SECURITY_DELAY").where(col("SECURITY_DELAY") > 0))

# COMMAND ----------

# MAGIC %md ### LATE_AIRCRAFT_DELAY

# COMMAND ----------

display(df_airlines.select("LATE_AIRCRAFT_DELAY").where(col("LATE_AIRCRAFT_DELAY") > 0))

# COMMAND ----------

# MAGIC %md ### DELAY REASONS

# COMMAND ----------

df_delay_reasons_carrier = df_airlines.where((col("CARRIER_DELAY") > 0) & (col("DEP_DEL15") == 1)).select(count("CARRIER_DELAY").alias("count")).withColumn("Reason", lit("Carrier"))
df_delay_reasons_weather = df_airlines.where((col("WEATHER_DELAY") > 0) & (col("DEP_DEL15") == 1)).select(count("WEATHER_DELAY").alias("count")).withColumn("Reason", lit("Weather"))
df_delay_reasons_nas = df_airlines.where((col("NAS_DELAY") > 0)  & (col("DEP_DEL15") == 1)).select(count("NAS_DELAY").alias("count")).withColumn("Reason", lit("Nas"))
df_delay_reasons_security = df_airlines.where((col("SECURITY_DELAY") > 0)  & (col("DEP_DEL15") == 1)).select(count("SECURITY_DELAY").alias("count")).withColumn("Reason", lit("Security"))
df_delay_reasons_aircraft = df_airlines.where((col("LATE_AIRCRAFT_DELAY") > 0) & (col("DEP_DEL15") == 1)).select(count("LATE_AIRCRAFT_DELAY").alias("count")).withColumn("Reason", lit("Late Aircraft"))

df_delay_reasons = reduce(DataFrame.unionAll, [df_delay_reasons_carrier, df_delay_reasons_weather, df_delay_reasons_nas, df_delay_reasons_security, df_delay_reasons_aircraft]) 

# COMMAND ----------

display(df_delay_reasons)

# COMMAND ----------

# MAGIC %md ## Table Cleanup

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Since we cannot use any of the departure performance features, flight summaries features, cause of delay features for the prediction and since some of the features such as carrier features, origin or destination airport features have redunant features which do not add value to prediction, our choice of features to use would be - Flight Number, Airline, Origin Airport, Destination Airport, Scheduled Flight Departure Time and the output/target variable DEP_DEL15.
# MAGIC 
# MAGIC All other prediction variables would come from weather table.

# COMMAND ----------

# MAGIC %md ### Timezone conversion

# COMMAND ----------

# MAGIC %md As the temporal features in the airlines tables is in local or airport timezones whereas the weather data is in UTC timezone, we convert the temporal features in the airlines table to UTC so as to make joins easier.

# COMMAND ----------

# DBTITLE 1,Timezone conversion functions
#Helper methods for timezone conversion

import airporttime
from datetime import datetime
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType, TimestampType
import math

def convert_airport_to_utc(airport, str_date, departure_time):
  apt = airporttime.AirportTime(iata_code=airport)
  
  if departure_time is None:
    return None
  
  departure_hour = math.floor(departure_time/100)
  departure_minute = departure_time%100
  date = int(str_date[8:10])
  month = int(str_date[5:7])
  year = int(str_date[0:4])
  
  if departure_hour >= 24:
    departure_hour = departure_hour - 24
    date = date + 1
    
  if month in [1, 3, 5, 7, 8, 10, 12] and date > 31:
    date = 1
    month = month + 1
  
  if month in [4, 6, 9, 11] and date > 30:
    date = 1
    month = month + 1
  
  if month == 2 and date > 28:
    date = 1
    month = month + 1
  
  if month > 12:
    month = 1
    year = year + 1
  
  local_time = datetime(year, month, date, departure_hour, departure_minute)
  utc_time = apt.to_utc(local_time)
  return utc_time

def convert_airport_to_utc_date(airport, date, departure_time):
  utc = convert_airport_to_utc(airport, date, departure_time)
  return str(utc.date())

def convert_airport_to_utc_hour(airport, date, departure_time):
  utc = convert_airport_to_utc(airport, date, departure_time)
  return utc.hour

convert_airport_to_utc_udf = udf(convert_airport_to_utc, TimestampType())
convert_airport_to_utc_date_udf = udf(convert_airport_to_utc_date)
convert_airport_to_utc_hour_udf = udf(convert_airport_to_utc_hour, IntegerType())

# COMMAND ----------

# DBTITLE 1,Test Cases for Timezone conversion
#testing timezone conversion

print(convert_airport_to_utc("JFK", "2022-03-18", 730))
print(convert_airport_to_utc_date("JFK", "2022-03-18", 1040))
print(convert_airport_to_utc_hour("JFK", "2022-03-18", 1432))
print(convert_airport_to_utc("JFK", "2022-12-31", 2430))

# COMMAND ----------

from pyspark.sql.functions import col, floor

def convert_timezone(dataframe):
  converted_dataframe = dataframe \
                                .withColumn("FL_DATE_UTC", convert_airport_to_utc_date_udf(dataframe.ORIGIN, dataframe.FL_DATE, dataframe.CRS_DEP_TIME)) \
                                .withColumn("DEP_HOUR_UTC", convert_airport_to_utc_hour_udf(dataframe.ORIGIN, dataframe.FL_DATE, dataframe.CRS_DEP_TIME)) \
                                .withColumn("FL_DATETIME_UTC", convert_airport_to_utc_udf(dataframe.ORIGIN, dataframe.FL_DATE, dataframe.CRS_DEP_TIME)) \
                                .withColumn("ARR_DATETIME_UTC", when( dataframe.ARR_TIME.isNull(), lit(None))
                                            .otherwise(convert_airport_to_utc_udf(dataframe.DEST, dataframe.FL_DATE, dataframe.CRS_ARR_TIME))) \
                                .withColumn("DEP_HOUR", floor(dataframe.CRS_DEP_TIME/100))
  return converted_dataframe




# COMMAND ----------

df_airlines = convert_timezone(df_airlines)

# COMMAND ----------

display(df_airlines)

# COMMAND ----------

# MAGIC %md ### Compute Number and percentage of departure delays in the last N hours for Airport

# COMMAND ----------

# DBTITLE 1,Query to get the number and percentage of departure delays in the last N hours for airport
from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number, sum, lit

def compute_departure_delays(dataframe):
  dataframe.createOrReplaceTempView("airlines")
  
  df_updated = spark.sql("""SELECT *, 
    
    SUM(CASE WHEN DEP_DEL15 == 1 THEN 1 ELSE 0 END) OVER (
        PARTITION BY origin
        ORDER BY fl_datetime_utc 
        RANGE BETWEEN INTERVAL 2 HOURS PRECEDING AND CURRENT ROW
     ) AS number_dep_delay_origin_2h,

    SUM(CASE WHEN DEP_DEL15 == 1 THEN 1 ELSE 0 END) OVER (
        PARTITION BY origin
        ORDER BY fl_datetime_utc 
        RANGE BETWEEN INTERVAL 4 HOURS PRECEDING AND CURRENT ROW
     ) AS number_dep_delay_origin_4h,
     
     count(DEP_DEL15) OVER (
        PARTITION BY origin 
        ORDER BY fl_datetime_utc 
        RANGE BETWEEN INTERVAL 2 HOURS PRECEDING AND CURRENT ROW
     ) AS number_flights_origin_2h,

    count(DEP_DEL15) OVER (
        PARTITION BY origin 
        ORDER BY fl_datetime_utc 
        RANGE BETWEEN INTERVAL 4 HOURS PRECEDING AND CURRENT ROW
     ) AS number_flights_origin_4h
     
     FROM airlines""")
  
  df_updated = df_updated \
                        .withColumn("number_dep_delay_origin", col("number_dep_delay_origin_4h") - col("number_dep_delay_origin_2h")) \
                        .withColumn("number_flights_origin", col("number_flights_origin_4h") - col("number_flights_origin_2h")) \
                        .drop("number_dep_delay_origin_4h", "number_dep_delay_origin_2h", "number_flights_origin_4h", "number_flights_origin_2h")

  return df_updated

# COMMAND ----------

df_airlines = compute_departure_delays(df_airlines)

# COMMAND ----------

display(df_airlines.where( (col("ORIGIN") == 'ATL') & (col('FL_DATE_UTC') == '2015-01-01') ).sort("FL_DATETIME_UTC"))

# COMMAND ----------

# MAGIC %md ### Compute Number and percentage of departure delays in the last N hours for Carrier

# COMMAND ----------

# DBTITLE 1,Query to get the number and percentage of departure delays in the last N hours for airport
from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number, sum, lit

def compute_carrier_delays(dataframe):
  dataframe.createOrReplaceTempView("airlines")
  
  df_updated = spark.sql("""SELECT *, 
    
    SUM(CASE WHEN DEP_DEL15 == 1 THEN 1 ELSE 0 END) OVER (
        PARTITION BY op_carrier
        ORDER BY fl_datetime_utc 
        RANGE BETWEEN INTERVAL 2 HOURS PRECEDING AND CURRENT ROW
     ) AS number_dep_delay_carrier_2h,

    SUM(CASE WHEN DEP_DEL15 == 1 THEN 1 ELSE 0 END) OVER (
        PARTITION BY op_carrier
        ORDER BY fl_datetime_utc 
        RANGE BETWEEN INTERVAL 4 HOURS PRECEDING AND CURRENT ROW
     ) AS number_dep_delay_carrier_4h,
     
     count(DEP_DEL15) OVER (
        PARTITION BY op_carrier 
        ORDER BY fl_datetime_utc 
        RANGE BETWEEN INTERVAL 2 HOURS PRECEDING AND CURRENT ROW
     ) AS number_flights_carrier_2h,

    count(DEP_DEL15) OVER (
        PARTITION BY op_carrier 
        ORDER BY fl_datetime_utc 
        RANGE BETWEEN INTERVAL 4 HOURS PRECEDING AND CURRENT ROW
     ) AS number_flights_carrier_4h
     
     FROM airlines""")
  
  df_updated = df_updated \
                        .withColumn("number_dep_delay_carrier", col("number_dep_delay_carrier_4h") - col("number_dep_delay_carrier_2h")) \
                        .withColumn("number_flights_carrier", col("number_flights_carrier_4h") - col("number_flights_carrier_2h")) \
                        .drop("number_dep_delay_carrier_4h", "number_dep_delay_carrier_2h", "number_flights_carrier_4h", "number_flights_carrier_2h")

  return df_updated

# COMMAND ----------

df_airlines = compute_carrier_delays(df_airlines)

# COMMAND ----------

display(df_airlines.where( (col("ORIGIN") == 'ATL') & (col('FL_DATE_UTC') == '2015-01-01') ).sort("FL_DATETIME_UTC"))

# COMMAND ----------

# MAGIC %md ### Number and percentage arrival delays in the last N hours for the Tail Number

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number, sum, lit

def compute_arrival_delays(dataframe):
  dataframe.createOrReplaceTempView("airlines")
  
  df_updated = spark.sql("""SELECT *, 
    
    SUM(CASE WHEN ARR_DEL15 == 1 THEN 1 ELSE 0 END) OVER (
        PARTITION BY tail_num
        ORDER BY fl_datetime_utc 
        RANGE BETWEEN INTERVAL 2 HOURS PRECEDING AND CURRENT ROW
     ) AS number_arr_delay_tail_2h,
     
     SUM(CASE WHEN ARR_DEL15 == 1 THEN 1 ELSE 0 END) OVER (
        PARTITION BY tail_num
        ORDER BY fl_datetime_utc 
        RANGE BETWEEN INTERVAL 4 HOURS PRECEDING AND CURRENT ROW
     ) AS number_arr_delay_tail_4h,
     
     count(ARR_DEL15) OVER (
        PARTITION BY tail_num 
        ORDER BY fl_datetime_utc 
        RANGE BETWEEN INTERVAL 2 HOURS PRECEDING AND CURRENT ROW
     ) AS number_flights_tail_2h,
     
     count(ARR_DEL15) OVER (
        PARTITION BY tail_num 
        ORDER BY fl_datetime_utc 
        RANGE BETWEEN INTERVAL 4 HOURS PRECEDING AND CURRENT ROW
     ) AS number_flights_tail_4h
     
     FROM airlines""")
  
  df_updated = df_updated \
                        .withColumn("number_arr_delay_tail", col("number_arr_delay_tail_4h") - col("number_arr_delay_tail_2h")) \
                        .withColumn("number_flights_tail", col("number_flights_tail_4h") - col("number_flights_tail_2h")) \
                        .drop("number_arr_delay_tail_4h", "number_arr_delay_tail_2h", "number_flights_tail_4h", "number_flights_tail_2h")
  
  return df_updated

# COMMAND ----------

df_airlines = compute_arrival_delays(df_airlines)

# COMMAND ----------

display(df_airlines.where( (col("TAIL_NUM") == 'N025AA') & (col('FL_DATE_UTC') == '2015-01-01') ).sort("FL_DATETIME_UTC"))

# COMMAND ----------

# MAGIC %md ### Previous Leg Arrival Time of Aircraft

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number, sum, lit

def compute_previous_arrival_time(dataframe):
  window = Window.partitionBy("TAIL_NUM").orderBy("FL_DATETIME_UTC")
  df_updated = dataframe.withColumn("previous_flight_arrival_time", when(col('ARR_DATETIME_UTC').isNull(), lit(None))
                                    .otherwise(lag("ARR_DATETIME_UTC", 1).over(window)))
    
  return df_updated

# COMMAND ----------

df_airlines = compute_previous_arrival_time(df_airlines)

# COMMAND ----------

display(df_airlines)

# COMMAND ----------

display(df_airlines.where( (col("TAIL_NUM") == 'N025AA') & (col('FL_DATE_UTC') == '2015-01-01') ).sort("FL_DATETIME_UTC"))

# COMMAND ----------

display(df_airlines.where((df_airlines.FL_Date <= "2019-03-04") 
                               & (df_airlines.FL_Date >= "2019-03-03")
                               & (df_airlines.TAIL_NUM == "N554NN") \
                               ).select("FL_DATETIME_UTC", "previous_flight_arrival_time", 
                                        "TAIL_NUM", 
                                        "DISTANCE", "ORIGIN", "DEST", 
                                        "*") \
                                .sort("FL_DATETIME_UTC"))

# COMMAND ----------

# MAGIC %md ### Cleanup

# COMMAND ----------

# DBTITLE 1,Function to cleanup airlines dataset
#cleaning for flights table
def clean_flights_table(dataframe):
  needed_columns = ["FL_DATE_UTC", "DEP_HOUR_UTC", "FL_DATETIME_UTC", "OP_CARRIER_AIRLINE_ID", "OP_CARRIER_FL_NUM", "TAIL_NUM", "ORIGIN", "Origin_State_Abr", "DEST", "Dest_State_Abr", "DEP_DEL15", "DEP_HOUR", "ARR_DEL15", "CRS_ARR_TIME", "ARR_TIME", "Year", "FL_Date", "CRS_DEP_TIME", "DEP_TIME", "Day_Of_Week", "Month", "Distance", "AIR_TIME", "number_dep_delay_origin", "number_flights_origin", "number_arr_delay_tail", "number_flights_tail", "percent_dep_delay_origin", "percent_arr_delay_tail", "number_dep_delay_carrier", "number_flights_carrier", "percent_dep_delay_carrier", "previous_flight_arrival_time", "resting_time_tail_num", "NAS_DELAY", "CARRIER_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY","SECURITY_DELAY"]

  cleaned_dataframe = dataframe.where(dataframe.CANCELLED == 0)\
                                  .withColumn("percent_dep_delay_origin", col('number_dep_delay_origin')*100/col('number_flights_origin')) \
                                  .withColumn("percent_arr_delay_tail", col('number_arr_delay_tail')*100/col('number_flights_tail')) \
                                  .withColumn("percent_dep_delay_carrier", col('number_dep_delay_carrier')*100/col('number_flights_carrier')) \
                                  .withColumn("resting_time_tail_num_secs", 
                                              when(col('previous_flight_arrival_time').isNull(), lit(None))
                                              .otherwise(col("FL_DATETIME_UTC").cast("long") - col('previous_flight_arrival_time').cast("long"))) \
                                  .withColumn("resting_time_tail_num", col("resting_time_tail_num_secs")/60) \
                                  .select([col for col in needed_columns])
  return cleaned_dataframe

# COMMAND ----------

# DBTITLE 1,Clean the airlines dataset
#clean dataset
df_airlines_clean = clean_flights_table(df_airlines)

# COMMAND ----------

df_airlines_clean.printSchema()

# COMMAND ----------

# MAGIC %md ### Save the cleaned datasets

# COMMAND ----------

# DBTITLE 1,Save the cleaned up airlines datasets
#Write to blob

df_airlines_clean.write.mode('overwrite').parquet(f"{blob_url}/airline_data_clean")

# COMMAND ----------

display(df_airlines_clean.where((df_airlines_clean.FL_DATE_UTC == '2015-03-04') & 
                                (df_airlines_clean.DEP_HOUR_UTC == 20) & 
                                (df_airlines_clean.OP_CARRIER_FL_NUM == '44')))

# COMMAND ----------

display(df_airlines.where((col('FL_DATE') == '2015-03-04') & 
                          (col('OP_CARRIER_FL_NUM') == '44') & 
                          (col('ORIGIN') == 'ATL')))

# COMMAND ----------

df_airlines_clean.printSchema()

# COMMAND ----------

display(df_airlines_clean)

# COMMAND ----------

# MAGIC %md ## Load the cleaned Dataset

# COMMAND ----------

df_airlines_clean = spark.read.parquet(f"{blob_url}/airline_data_clean")

# COMMAND ----------

display(df_airlines_clean
        .withColumn("ANY_PRIOR_DELAYS", when(col("number_dep_delay_origin") > 0, 1).otherwise(0))
        .select("DEP_DEL15", "ANY_PRIOR_DELAYS")
        .groupBy("DEP_DEL15", "ANY_PRIOR_DELAYS")
        .count())

# COMMAND ----------

display(df_airlines_clean
        .withColumn("ANY_PRIOR_ARR_DELAYS", when(col("number_arr_delay_tail") > 0, 1).otherwise(0))
        .select("DEP_DEL15", "ANY_PRIOR_ARR_DELAYS")
        .groupBy("DEP_DEL15", "ANY_PRIOR_ARR_DELAYS")
        .count())

# COMMAND ----------

display(df_airlines_clean.where((df_airlines_clean.FL_Date <= "2019-03-04") 
                               & (df_airlines_clean.FL_Date >= "2019-03-03")
                               & (df_airlines_clean.TAIL_NUM == "N554NN") \
                               ).select("FL_DATETIME_UTC", "previous_flight_arrival_time", 
                                        "TAIL_NUM", 
                                        "DISTANCE", "ORIGIN", "DEST", 
                                        "*") \
                                .sort("FL_DATETIME_UTC"))

# COMMAND ----------

display(df_airlines_clean
        .withColumn("ANY_PRIOR_CARRIER_DELAYS", when(col("number_dep_delay_carrier") > 0, 1).otherwise(0))
        .select("DEP_DEL15", "ANY_PRIOR_CARRIER_DELAYS")
        .groupBy("DEP_DEL15", "ANY_PRIOR_CARRIER_DELAYS")
        .count())

# COMMAND ----------

display(df_airlines_clean
        .groupBy("DEP_DEL15")
        .mean("resting_time_tail_num"))