# Databricks notebook source
# MAGIC %md
# MAGIC ## EDA for Airline Data

# COMMAND ----------

from pyspark.sql.functions import col,split,isnan, when, count, concat, lit, desc, floor
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
display(dbutils.fs.ls("/mnt/mids-w261/datasets_final_project"))

# COMMAND ----------

# DBTITLE 1,Load the airlines data
df_airlines_3m = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data_3m/")
df_airlines_6m = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data_6m/")

# COMMAND ----------

display(df_airlines_3m)

# COMMAND ----------

df_airlines_3m.printSchema()

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
summary = df_airlines_3m.summary()

# COMMAND ----------

display(summary)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC From the summary table above, we see that that departure performance features, DEP_TIME, DEP_DELAY, DEP_DELAY_NEW, DEP_DEL15, DEP_DELAY_GROUP, TAXI_OUT, WHEELS_OFF do not have values for all entries. Among the arrival performance features, ARR_TIME, ARR_DELAY, ARR_DELAY_NEW, ARR_DEL15, ARR_DELAY_GROUP do not have values for all entries. In the Flight Summaries group, ACTUAL_ELAPSED_TIME, AIR_TIME are missing values for some records. Understandably, the cause of delay group features, CARRIER_DELAY, WEATHER_DELAY, NAS_DELAY, SECURITY_DELAY, LATE_AIRCRAFT_DELAY have values for only a few records. In the Gate Return Information feature group, FIRST_DEP_TIME, TOTAL_ADD_GTIME, LONGEST_ADD_GTIME is available only for a few records. In the Diverted Airport Information feature group, majority of these features have no values indicating that a very small fraction of flights were diverted.

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

# MAGIC %md ### Month
# MAGIC 
# MAGIC We see that the airlines data seems to be spread equally across the 3 months.

# COMMAND ----------

display(df_airlines_3m.select("Month").groupBy("Month").count().sort(desc("count")))

# COMMAND ----------

# MAGIC %md ### Day of Month
# MAGIC 
# MAGIC We see that the airlines data seems to be spread equally across all days of the month except around the first day of the month which shows fewer flights.

# COMMAND ----------

display(df_airlines_3m.select("Day_Of_Month").groupBy("Day_Of_Month").count())

# COMMAND ----------

# MAGIC %md ### Day of Week
# MAGIC 
# MAGIC We see that the airlines data seems to be spread equally across all days of the week

# COMMAND ----------

display(df_airlines_3m.select("Day_Of_Week").groupBy("Day_Of_Week").count())

# COMMAND ----------

# MAGIC %md ### Op_Carrier
# MAGIC 
# MAGIC For the 3 month airlines data, there are a few operators such as DL, EV, MQ who have more flights than other operators

# COMMAND ----------

display(df_airlines_3m.select("Op_Carrier").groupBy("Op_Carrier").count().sort(desc("count")))

# COMMAND ----------

# MAGIC %md ### Op_Carrier_Fl_Num
# MAGIC 
# MAGIC For the 3 month airlines data, there are a few operators such as DL, EV, MQ who have more flights than other operators

# COMMAND ----------

display(df_airlines_3m.select("Op_Carrier_Fl_Num").groupBy("Op_Carrier_Fl_Num").count().sort(desc("count")))

# COMMAND ----------

# MAGIC %md ### Origin-Destination
# MAGIC 
# MAGIC We see that there are some origin-destination airport pairs where the air traffic or number of flights is highest such as ORD-LGA, ATL-LGA

# COMMAND ----------

display(df_airlines_3m.withColumn("Origin-Dest", concat(col("Origin"), lit("-"), col("Dest"))).select("Origin-Dest").groupBy("Origin-Dest").count().sort(desc("count")))

# COMMAND ----------

# MAGIC %md ### DEP_DEL15
# MAGIC 
# MAGIC Around 4826 records in the 3 months airline data do not have any departure delay status.

# COMMAND ----------

display(df_airlines_3m.select("DEP_DEL15").groupBy("DEP_DEL15").count())

# COMMAND ----------

# MAGIC %md ### CANCELLATION_CODE
# MAGIC 
# MAGIC Around 156073 records in the 3 months airline data do not have any cancellation code, since only a small percentage of flights would be cancelled. However, we also notice that CANCELLATION_CODE **"B"** has more entries that the other two code "A" and "C".

# COMMAND ----------

display(df_airlines_3m.select("CANCELLATION_CODE").groupBy("CANCELLATION_CODE").count())

# COMMAND ----------

# MAGIC %md ### DEP_DEL15 and CANCELLATION_CODE

# COMMAND ----------

display(df_airlines_3m.select("DEP_DEL15", "CANCELLATION_CODE").groupBy("DEP_DEL15", "CANCELLATION_CODE").count())

# COMMAND ----------

# MAGIC %md ### DIV_AIRPORT_LANDINGS
# MAGIC 
# MAGIC While a majority of the flights have 0 airport diversions, among the diverted airport landings, the most number of diverted airport landings is 1. 

# COMMAND ----------

display(df_airlines_3m.select("DIV_AIRPORT_LANDINGS").groupBy("DIV_AIRPORT_LANDINGS").count())

# COMMAND ----------

# MAGIC %md ## Numerical Variables

# COMMAND ----------

# MAGIC %md ### CARRIER_DELAY
# MAGIC 
# MAGIC We see most carrier delays are short duration.

# COMMAND ----------

display(df_airlines_3m.select("CARRIER_DELAY").where(col("CARRIER_DELAY") > 0))

# COMMAND ----------

# MAGIC %md ### WEATHER_DELAY

# COMMAND ----------

display(df_airlines_3m.select("WEATHER_DELAY").where(col("WEATHER_DELAY") > 0))

# COMMAND ----------

, NAS_DELAY, SECURITY_DELAY, LATE_AIRCRAFT_DELAY

# COMMAND ----------

# MAGIC %md ### NAS_DELAY

# COMMAND ----------

display(df_airlines_3m.select("NAS_DELAY").where(col("NAS_DELAY") > 0))

# COMMAND ----------

# MAGIC %md ### SECURITY_DELAY

# COMMAND ----------

display(df_airlines_3m.select("SECURITY_DELAY").where(col("SECURITY_DELAY") > 0))

# COMMAND ----------

# MAGIC %md ### LATE_AIRCRAFT_DELAY

# COMMAND ----------

display(df_airlines_3m.select("LATE_AIRCRAFT_DELAY").where(col("LATE_AIRCRAFT_DELAY") > 0))

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

def convert_airport_to_utc(airport, date, departure_time): 
  apt = airporttime.AirportTime(iata_code=airport)
  local_time = datetime(int(date[0:4]), int(date[5:7]), int(date[8:10]), math.floor(departure_time/100), departure_time%100)
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

# COMMAND ----------

from pyspark.sql.functions import col, floor

def convert_timezone(dataframe):
  converted_dataframe = dataframe \
                                .withColumn("FL_DATE_UTC", convert_airport_to_utc_date_udf(dataframe.ORIGIN,dataframe.FL_DATE, dataframe.CRS_DEP_TIME)) \
                                .withColumn("DEP_HOUR_UTC", convert_airport_to_utc_hour_udf(dataframe.ORIGIN,dataframe.FL_DATE, dataframe.CRS_DEP_TIME)) \
                                .withColumn("FL_DATETIME_UTC", convert_airport_to_utc_udf(dataframe.ORIGIN,dataframe.FL_DATE, dataframe.CRS_DEP_TIME)) \
                                .withColumn("DEP_HOUR", floor(dataframe.CRS_DEP_TIME/100))
  return converted_dataframe



# COMMAND ----------

df_airlines_3m = convert_timezone(df_airlines_3m)
df_airlines_6m = convert_timezone(df_airlines_6m)

# COMMAND ----------

# MAGIC %md ### Compute Number and percentage of departure delays in the last N hours for Airport

# COMMAND ----------

# DBTITLE 1,Query to get the number and percentage of departure delays in the last N hours for airport
from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number, sum, lit

def compute_departure_delays(dataframe):
  dataframe.createOrReplaceTempView("airlines")
  
  df_updated = spark.sql("""SELECT *, SUM(CASE WHEN DEP_DEL15 == 1 THEN 1 ELSE 0 END) OVER (
        PARTITION BY origin
        ORDER BY fl_datetime_utc 
        RANGE BETWEEN INTERVAL 2 HOURS PRECEDING AND CURRENT ROW
     ) AS number_dep_delay_origin,
     count(DEP_DEL15) OVER (
        PARTITION BY origin 
        ORDER BY fl_datetime_utc 
        RANGE BETWEEN INTERVAL 2 HOURS PRECEDING AND CURRENT ROW
     ) AS number_flights_origin
     FROM airlines""")
  
  return df_updated

# COMMAND ----------

df_airlines_3m = compute_departure_delays(df_airlines_3m)

# COMMAND ----------

display(df_airlines_3m.where( (col("ORIGIN") == 'ATL') & (col('FL_DATE_UTC') == '2015-01-01') ).sort("FL_DATETIME_UTC"))

# COMMAND ----------

df_airlines_6m = compute_departure_delays(df_airlines_6m)

# COMMAND ----------

# MAGIC %md ### Number and percentage arrival delays in the last N hours for the Tail Number

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number, sum, lit

def compute_arrival_delays(dataframe):
  dataframe.createOrReplaceTempView("airlines")
  
  df_updated = spark.sql("""SELECT *, SUM(CASE WHEN ARR_DEL15 == 1 THEN 1 ELSE 0 END) OVER (
        PARTITION BY tail_num
        ORDER BY fl_datetime_utc 
        RANGE BETWEEN INTERVAL 2 HOURS PRECEDING AND CURRENT ROW
     ) AS number_arr_delay_tail,
     count(ARR_DEL15) OVER (
        PARTITION BY origin 
        ORDER BY fl_datetime_utc 
        RANGE BETWEEN INTERVAL 2 HOURS PRECEDING AND CURRENT ROW
     ) AS number_flights_tail
     FROM airlines""")
  return df_updated

# COMMAND ----------

df_airlines_3m = compute_arrival_delays(df_airlines_3m)
df_airlines_6m = compute_arrival_delays(df_airlines_6m)

# COMMAND ----------

display(df_airlines_3m.where( (col("TAIL_NUM") == 'N025AA') & (col('FL_DATE_UTC') == '2015-01-01') ).sort("FL_DATETIME_UTC"))

# COMMAND ----------

# MAGIC %md ### Cleanup

# COMMAND ----------

# DBTITLE 1,Function to cleanup airlines dataset
#cleaning for flights table
def clean_flights_table(dataframe):
  needed_columns = ["FL_DATE_UTC", "DEP_HOUR_UTC", "FL_DATETIME_UTC", "OP_CARRIER_AIRLINE_ID", "OP_CARRIER_FL_NUM", "ORIGIN", "DEST", "DEP_DEL15", "DEP_HOUR", "Day_Of_Week", "Month", "Distance", "number_dep_delay_origin", "number_flights_origin", "number_arr_delay_tail", "number_flights_tail", "percent_dep_delay_origin", "percent_arr_delay_tail"]

  cleaned_dataframe = dataframe.where(dataframe.CANCELLED == 0)\
                                  .withColumn("percent_dep_delay_origin", col('number_dep_delay_origin')*100/col('number_flights_origin')) \
                                  .withColumn("percent_arr_delay_tail", col('number_arr_delay_tail')*100/col('number_flights_tail')) \
                                  .select([col for col in needed_columns])

  
  return cleaned_dataframe

# COMMAND ----------

# DBTITLE 1,clean the 3 months airlines dataset
#clean 3m dataset
df_airlines_clean_3m = clean_flights_table(df_airlines_3m)#clean

display(df_airlines_clean_3m)

# COMMAND ----------

# DBTITLE 1,Clean the 6 months airlines dataset
#clean 6m dataset
df_airlines_clean_6m = clean_flights_table(df_airlines_6m)

display(df_airlines_clean_6m)

# COMMAND ----------

# MAGIC %md ### Save the cleaned datasets

# COMMAND ----------

# DBTITLE 1,Save the cleaned up airlines datasets
#Write to blob

df_airlines_clean_3m.write.mode("overwrite").parquet(f"{blob_url}/airline_data_clean_3m")
df_airlines_clean_6m.write.mode("overwrite").parquet(f"{blob_url}/airline_data_clean_6m")

# COMMAND ----------

display(df_airlines_clean_3m)