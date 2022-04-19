# Databricks notebook source
# MAGIC %md
# MAGIC # Scratch workbook to build new features

# COMMAND ----------

# MAGIC %md
# MAGIC Using 3m data to develop the feature. Will need to be run on the full joined dataset when ready

# COMMAND ----------

df_airlines_3m = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data_3m/")

# COMMAND ----------

#cleaning for flights table
import airporttime
from datetime import datetime
from pyspark.sql.functions import udf, sum
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

def clean_flights_table(dataframe):
  needed_columns = ["FL_DATE_UTC", "DEP_HOUR_UTC", "FL_DATETIME_UTC", "OP_CARRIER_AIRLINE_ID", "OP_CARRIER_FL_NUM", "ORIGIN", "DEST", "DEP_DEL15"]

  cleaned_dataframe = dataframe.where(dataframe.CANCELLED == 0)\
                                  .withColumn("FL_DATE_UTC", convert_airport_to_utc_date_udf(dataframe.ORIGIN,dataframe.FL_DATE, dataframe.CRS_DEP_TIME)) \
                                  .withColumn("DEP_HOUR_UTC", convert_airport_to_utc_hour_udf(dataframe.ORIGIN,dataframe.FL_DATE, dataframe.CRS_DEP_TIME)) \
                                  .withColumn("FL_DATETIME_UTC", convert_airport_to_utc_udf(dataframe.ORIGIN,dataframe.FL_DATE, dataframe.CRS_DEP_TIME)) \
                                  .select([col for col in needed_columns])

  
  return cleaned_dataframe

# COMMAND ----------

df_airlines_clean_3m = clean_flights_table(df_airlines_3m)#clean

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Number of flights delayed
# MAGIC 
# MAGIC This feature will count the number of flights that are delayed in the day and add that to the row. It will take into account how many hours before departure the prediction needs to be made. 

# COMMAND ----------

def determine_delayed_flights(dataframe, window):
    

# COMMAND ----------

df_airlines_clean_3m.printSchema()

# COMMAND ----------

display(df_airlines_clean_3m.where((df_airlines_clean_3m.FL_DATE_UTC == '2015-02-01') & (df_airlines_clean_3m.DEP_DEL15 == 1)).orderBy(df_airlines_clean_3m.DEP_HOUR_UTC))

# COMMAND ----------

print(df_airlines_clean_3m.select(df_airlines_clean_3m.FL_DATE_UTC, df_airlines_clean_3m.DEP_HOUR_UTC, df_airlines_clean_3m.DEP_DEL15).rdd.map(lambda x: (x[0], x[1], x[2])).collect()[0:5])


# COMMAND ----------

