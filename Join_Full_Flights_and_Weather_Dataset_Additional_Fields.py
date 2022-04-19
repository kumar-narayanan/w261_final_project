# Databricks notebook source
from pyspark.sql.functions import col,split,isnan, when, count, concat, lit, desc, floor, max, unix_timestamp
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import SQLContext
from pyspark.sql import DataFrameNaFunctions

blob_container = "team08-container" # The name of your container created in https://portal.azure.com
storage_account = "team08" # The name of your Storage account created in https://portal.azure.com
secret_scope = "team08-scope" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "team08-key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# COMMAND ----------

# MAGIC %md ## Load the data

# COMMAND ----------

# MAGIC %md ### Airport Station Lookup

# COMMAND ----------

#Load airport station lookup table
df_airport_station_lookup = spark.read.parquet(f"{blob_url}/airport_station_lookup2")
display(df_airport_station_lookup)

# COMMAND ----------

# MAGIC %md ### Flights

# COMMAND ----------

#Load cleaned airlines data
df_airlines = spark.read.parquet(f"{blob_url}/airline_data_clean/")
display(df_airlines)
print(df_airlines.count())

# COMMAND ----------

# MAGIC %md ### Weather

# COMMAND ----------

#Load Weather Data
df_weather = spark.read.parquet(f"{blob_url}/df_weather")

# COMMAND ----------

# MAGIC %md ## Joins

# COMMAND ----------

# MAGIC %md ### Flights and Airport Station Lookup
# MAGIC 
# MAGIC Left join the flights dataset with the airport station lookup using the origin airport code to obtain the weather stations closest to the origin airport of a flight.

# COMMAND ----------

#Join stations and airlines table
df_airlines_station = df_airlines.join(df_airport_station_lookup, df_airlines.ORIGIN == df_airport_station_lookup.Code, "left")
display(df_airlines_station)
print(df_airlines_station.count())

# COMMAND ----------

# MAGIC %md ### Flights and Weather
# MAGIC 
# MAGIC Left Join the Flights dataset with the Weather data set by joining on the closest weather station to the origin airport and the weather station of the weather readings. Get the weather readings for a 2 hour weather window, 2 hours prior to the flight departure time. 

# COMMAND ----------

#function to join weather data
from datetime import timedelta

def join_airlines_weather(df_airlines, df_weather, hrs_prior_to_departure = 2,  weather_window = 2):
    """This function joins the flights and weather dataset to get weather during a window"""
    weather_window_start = (hrs_prior_to_departure + weather_window)*3600
    weather_window_end = hrs_prior_to_departure*3600
    df_airlines_weather = df_airlines.join(df_weather, 
                           (df_airlines.station_id == df_weather.STATION) & 
                           ((weather_window_end <= unix_timestamp(df_airlines.FL_DATETIME_UTC) - (unix_timestamp(df_weather.DATE))) & (weather_window_start >= unix_timestamp(df_airlines.FL_DATETIME_UTC) - unix_timestamp(df_weather.DATE))),
                          "left"
                          )
    return df_airlines_weather

# COMMAND ----------

#Join weather data
df_airlines_weather = join_airlines_weather(df_airlines_station, df_weather)

# COMMAND ----------

# MAGIC %md ## Sanity Checks

# COMMAND ----------

# MAGIC %md ### Weather Reading Counts
# MAGIC 
# MAGIC Each flight should have around 4 readings. Check those that have fewer than 4 weather readings.

# COMMAND ----------

# DBTITLE 1,Check for flights with fewer than 4 weather readings
#check for less than 4 weather readings
display(df_airlines_weather.groupBy("FL_DATE_UTC", "DEP_HOUR_UTC", "OP_CARRIER_AIRLINE_ID", "OP_CARRIER_FL_NUM", "ORIGIN").count().where(col('count') < 4))

# COMMAND ----------

# MAGIC %md Check that flights departing at midnight have atleast one weather reading

# COMMAND ----------

# DBTITLE 1,Check for flights departing at midnight
#check for midnight departure
display(df_airlines_weather.where( (col('FL_DATE_UTC') == '2015-05-17') & 
                                     (col('ORIGIN') == 'ATL') & 
                                     (col('DEP_HOUR_UTC') == 0) & 
                                     (col('OP_CARRIER_FL_NUM') == '4738')))

# COMMAND ----------

# MAGIC %md Check that flights departing at 1AM have atleast one weather reading

# COMMAND ----------

#Check for 1 am departure
display(df_airlines_weather.where( (col('FL_DATE_UTC') == '2015-05-15') & 
                                     (col('ORIGIN') == 'ATL') & 
                                     (col('DEP_HOUR_UTC') == 1) & 
                                     (col('OP_CARRIER_FL_NUM') == '1652')))

# COMMAND ----------

# MAGIC %md Check that flights departing at 2AM atleast have one weather reading

# COMMAND ----------

#check for 2 am departure
display(df_airlines_weather.where( (col('FL_DATE_UTC') == '2015-01-25') & 
                                     (col('ORIGIN') == 'ATL') & 
                                     (col('DEP_HOUR_UTC') == 2) & 
                                     (col('OP_CARRIER_FL_NUM') == '5610')))

# COMMAND ----------

# MAGIC %md ### Flights with Missing Weather Data

# COMMAND ----------

# DBTITLE 1,Check for flights having no weather data
#check for missing weather data
df_missing_weather = df_airlines_weather.where(col("STATION").isNull())
print(df_missing_weather.count())

# COMMAND ----------

display(df_missing_weather.limit(100))

# COMMAND ----------

df_airport_without_stations = df_missing_weather.select("Origin").distinct()

# COMMAND ----------

display(df_airport_without_stations)

# COMMAND ----------

# MAGIC %md ### All weather readings should be 2 hours prior to departure time and must be in the 2 hour window

# COMMAND ----------

#Sanity check
display(df_airlines_weather.where((df_airlines_weather.FL_DATE_UTC == '2015-03-04') & (df_airlines_weather.DEP_HOUR_UTC == 20) & (df_airlines_weather.OP_CARRIER_FL_NUM == '44')))

# COMMAND ----------

df_airlines_weather.printSchema()

# COMMAND ----------

df_airlines_weather.rdd.getNumPartitions()

# COMMAND ----------

# MAGIC %md ## Save the Joined Dataset

# COMMAND ----------

#Save joined dataset
df_airlines_weather.write.partitionBy("Date_Only").mode('overwrite').parquet(f"{blob_url}/df_airlines_weather_updated_additional")

# COMMAND ----------

# MAGIC %md ## Load the Joined Dataset

# COMMAND ----------

df_airlines_weather = spark.read.parquet(f"{blob_url}/df_airlines_weather_updated_additional")

# COMMAND ----------

df_airlines_weather.count()

# COMMAND ----------

df_airlines_weather.printSchema()