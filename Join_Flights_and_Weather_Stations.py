# Databricks notebook source
# MAGIC %md # Join Flights and Weather Stations

# COMMAND ----------

# DBTITLE 1,Init storage account
from pyspark.sql.functions import col, max, unix_timestamp

blob_container = "team08-container" # The name of your container created in https://portal.azure.com
storage_account = "team08" # The name of your Storage account created in https://portal.azure.com
secret_scope = "team08-scope" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "team08-key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# COMMAND ----------

# MAGIC %md ## Joins
# MAGIC 
# MAGIC In order to predict whether a flight would be delayed or not, it is essential to find the weather conditions during departure. To join the airlines data with the weather data, we need to find out the nearest weather station to the airport. Since the airlines dataset does not have the location coordinate of the airports, we employ a secondary dataset that provides the latitude and longitude for all the airports in our airlines dataset.

# COMMAND ----------

# MAGIC %md ### Load the datasets

# COMMAND ----------

# MAGIC %md #### Stations 

# COMMAND ----------

# DBTITLE 1,Load Stations Data
df_stations = spark.read.parquet("/mnt/mids-w261/datasets_final_project/stations_data/*")
display(df_stations)

# COMMAND ----------

# DBTITLE 1,Get weather at a station
df_weather_at_station = df_weather_3m.where("STATION = '3809099999' AND DATE = '2015-03-12'")

# COMMAND ----------

display(df_weather_at_station)

# COMMAND ----------

# MAGIC %md #### Airport Geolocation Lookup Data
# MAGIC 
# MAGIC Load the airport geolocation data which gives the geolocation coordinates for each US airport.

# COMMAND ----------

# DBTITLE 1,Load Airport Geolocation Data

################# Airport LAT-LNG codes new for all airports in the airlines data #################
# File location and type
file_location = "/FileStore/tables/airport_code_lat_lng_new.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df_airport_lat_lng = spark.read.format(file_type) \
                           .option("inferSchema", infer_schema) \
                           .option("header", first_row_is_header) \
                           .option("sep", delimiter) \
                           .load(file_location)

df_airport_lat_lng = df_airport_lat_lng.withColumnRenamed("Origin", "Code")
display(df_airport_lat_lng)

# COMMAND ----------

df_weather_at_station = df_weather.where(col('station') == 69002093218 )
display(df_weather_at_station)

# COMMAND ----------

# MAGIC %md ### Calculate Distance between Airports and Weather Stations
# MAGIC 
# MAGIC In order to find the closest weather station to an airport, we use the Haversine formula to compute the distance between the airports and the weather stations.

# COMMAND ----------

# MAGIC %md #### Haversine Distance

# COMMAND ----------

# DBTITLE 1,Function to calculate distance between two locations
from math import atan2, sin, cos, radians, sqrt

def getDistance(lat_deg1, long_deg1, lat_deg2, long_deg2, in_miles = True):
    '''This function gets the distance between two coordinates using Haversine formula'''
       
# Haversine formula: 
# a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2) 
# c = 2 ⋅ atan2( √a, √(1−a) )
# d = R ⋅ c
# where	φ is latitude, λ is longitude, R is earth’s radius (mean radius = 6,371km); note that angles need to be in radians to pass to trig functions!


#     First, convert the latitude and longitude values from decimal degrees to radians. For this divide the values of longitude and latitude of both the points by 180/pi.
    lat1, long1 = radians(lat_deg1), radians(long_deg1)
    lat2, long2 = radians(lat_deg2), radians(long_deg2)
    
    if in_miles:
        R = 3958.8
    else:
        R = 1.60934 * 3958.8
        
    dlat = lat1 - lat2
    dlong = long1 - long2
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlong/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    
    return distance

# COMMAND ----------

# DBTITLE 1,Create UDF to calculate distance
from pyspark.sql.types import FloatType
getDistance_udf = udf(getDistance, FloatType())

# COMMAND ----------

# DBTITLE 1,Test the distance function
getDistance(53.32055555555556, -1.7297222222222221, 53.31861111111111, -1.6997222222222223)

# COMMAND ----------

df_airport_lat_lng.printSchema()

# COMMAND ----------

#################################################
display(df_airport_lat_lng.select("Lat", "Lng", "Lat_Lng").where("Code = 'ATL'"))

# COMMAND ----------

atlanta = (33.6367, -84.428101)
getDistance(atlanta[0], atlanta[1], 33.63, -84.442)

# COMMAND ----------

# MAGIC %md #### Filter Stations
# MAGIC 
# MAGIC Since the Stations dataset contains multiple entries for a station, each entry providing distance to another neighboring station, we filter out only the stations where distance to the neighbor is zero.

# COMMAND ----------

# DBTITLE 1,Get all the stations which are neighbors to themselves.
df_stations_unique = df_stations.where(col("distance_to_neighbor") == 0)

# COMMAND ----------

display(df_stations_unique)

# COMMAND ----------

# MAGIC %md #### Get closest weather station to an airport

# COMMAND ----------

# DBTITLE 1,Function to get the closest station to an airport
from pyspark.sql.functions import col,lit
def closest_station(airport_lat, airport_long):
    '''function to compute the closest weather station to a given airport'''
    print(airport_lat, airport_long)
    df_closest_station = df_stations_unique.withColumn("distance_from_airport", getDistance_udf(lit(float(airport_lat)), lit(float(airport_long)), 'lat', 'lon')).orderBy("distance_from_airport")
    return df_closest_station.first()['station_id']
    

# COMMAND ----------

# DBTITLE 1,Create UDF to get closest station to an airport
from pyspark.sql.types import StringType
closest_station_udf = udf(closest_station, StringType())

# COMMAND ----------

closest_station('33.6367', '-84.428101')

# COMMAND ----------

# MAGIC %md ##### Cross Join Airport Geolocations Lookup and Stations
# MAGIC 
# MAGIC We cross join the airport geolocations lookup table and stations table to get the distance of the airports from the weather stations.

# COMMAND ----------

# DBTITLE 1,Cross Join Airports and Stations
df_airport_station = df_airport_lat_lng.withColumnRenamed('Lat', 'Latitude').crossJoin(df_stations_unique)

# COMMAND ----------

# DBTITLE 1,Get distance of stations from airports
df_airport_station_distance = df_airport_station.withColumn('distance_to_station', getDistance_udf('Latitude', 'Lng', 'lat', 'lon'))

# COMMAND ----------

display(df_airport_station_distance)

# COMMAND ----------

# MAGIC %md ##### Closest Weather stations
# MAGIC 
# MAGIC Partition the cross joined datasets by the airport and order by distance to the stations.

# COMMAND ----------

# DBTITLE 1,Partition by airport and get the closest station to the airport
from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number
windowCode = Window.partitionBy("Code").orderBy(col("distance_to_station"))
df_airport_station_closest = df_airport_station_distance.withColumn("row",row_number().over(windowCode)) \
  .filter(col("row") == 1).drop("row")

# COMMAND ----------

display(df_airport_station_closest)

# COMMAND ----------

# MAGIC %md ##### Save the Airport Station Lookup Data

# COMMAND ----------

# DBTITLE 1,Rename Columns
display(df_airport_station_closest.select('Code', 'usaf', 'wban', 'station_id', col('Latitude').alias('airport_lat'), col('Lng').alias('airport_long'), col('lat').alias('station_lat'), col('lon').alias('station_long'), 'distance_to_station'))

# COMMAND ----------

# DBTITLE 1,Save the airport_station_lookup table to parquet
df_airport_station_closest.select('Code', 'usaf', 'wban', 'station_id', col('Latitude').alias('airport_lat'), col('Lng').alias('airport_long'), col('lat').alias('station_lat'), col('lon').alias('station_long'), 'distance_to_station').write.mode("overwrite").parquet(f"{blob_url}/airport_station_lookup2")

# COMMAND ----------

# DBTITLE 1,Load airport_station_lookup from parquet
df_airport_station_lookup = spark.read.parquet(f"{blob_url}/airport_station_lookup2")

# COMMAND ----------

display(df_airport_station_lookup)

# COMMAND ----------

display(df_airport_station_lookup.where(col('Code') == 'ORD'))

# COMMAND ----------

df_weather_at_atlanta_on_day = df_weather.where("STATION = 72219013874 AND DATE > '2015-03-13' AND DATE < '2015-03-14'")

# COMMAND ----------

df_weather_at_atlanta_on_day.count()

# COMMAND ----------

display(df_weather_at_atlanta_on_day)

# COMMAND ----------

# MAGIC %md ### Join Weather and Flights datasets

# COMMAND ----------

# DBTITLE 1,Load Cleaned Airlines Data
df_airlines_3m = spark.read.parquet(f"{blob_url}/airline_data_clean3_3m")
df_airlines_6m = spark.read.parquet(f"{blob_url}/airline_data_clean3_6m")

# COMMAND ----------

display(df_airlines_3m)

# COMMAND ----------

display(df_airlines_3m.groupBy("FL_DATE_UTC", "DEP_HOUR_UTC", "ORIGIN").count().sort("FL_DATE_UTC"))

# COMMAND ----------

# MAGIC %md #### Join Flights and Airport Station Lookup data

# COMMAND ----------

# DBTITLE 1,Join the airlines and airport_station_lookup data
df_airlines_station_3m = df_airlines_3m.join(df_airport_station_lookup, df_airlines_3m.ORIGIN == df_airport_station_lookup.Code, "left")

# COMMAND ----------

display(df_airlines_station_3m)

# COMMAND ----------

df_airlines_station_3m.write.mode('overwrite').parquet(f"{blob_url}/df_airlines_station_3m")

# COMMAND ----------

df_airlines_station_6m = df_airlines_6m.join(df_airport_station_lookup, df_airlines_6m.ORIGIN == df_airport_station_lookup.Code, "left")

# COMMAND ----------

df_airlines_station_6m.write.mode('overwrite').parquet(f"{blob_url}/df_airlines_station_6m")

# COMMAND ----------

# MAGIC %md #### Join on Weather data

# COMMAND ----------

# DBTITLE 1,Load Cleaned Weather Data
df_weather_3m = spark.read.parquet(f"{blob_url}/df_weather_3m")
df_weather_6m = spark.read.parquet(f"{blob_url}/df_weather_6m")

# COMMAND ----------

display(df_weather_3m)

# COMMAND ----------

df_weather_3m.printSchema()

# COMMAND ----------

# DBTITLE 1,Create a test Airlines dataset
from datetime import datetime 

test_airlines_data = [
    {'Origin': 'ORD', 'station_id': '1', 'FL_DATETIME_UTC': datetime(2022, 3, 19, 3, 30), 'Flight': '1', 'Carier': 'X'}, 
    {'Origin': 'ATL', 'station_id': '2', 'FL_DATETIME_UTC': datetime(2022, 3, 19, 3, 40), 'Flight': '2', 'Carier': 'Y'}, 
    {'Origin': 'ORD', 'station_id': '1', 'FL_DATETIME_UTC': datetime(2022, 3, 19, 3, 40), 'Flight': '3', 'Carier': 'Z'}, 
    {'Origin': 'ORD', 'station_id': '1', 'FL_DATETIME_UTC': datetime(2022, 3, 19, 3, 50), 'Flight': '4', 'Carier': 'X'}, 
    {'Origin': 'ATL', 'station_id': '2', 'FL_DATETIME_UTC': datetime(2022, 3, 19, 4, 20), 'Flight': '5', 'Carier': 'Y'}, 
    {'Origin': 'ORD', 'station_id': '1', 'FL_DATETIME_UTC': datetime(2022, 3, 19, 4, 30), 'Flight': '6', 'Carier': 'Z'}, 
    {'Origin': 'ORD', 'station_id': '1', 'FL_DATETIME_UTC': datetime(2022, 3, 19, 5, 15), 'Flight': '7', 'Carier': 'X'}, 
    {'Origin': 'ORD', 'station_id': '1', 'FL_DATETIME_UTC': datetime(2022, 3, 19, 5, 30), 'Flight': '8', 'Carier': 'Y'}, 
    {'Origin': 'ATL', 'station_id': '2', 'FL_DATETIME_UTC': datetime(2022, 3, 19, 5, 45), 'Flight': '9', 'Carier': 'Z'},
    {'Origin': 'JFK', 'station_id': '3', 'FL_DATETIME_UTC': datetime(2022, 3, 19, 5, 30), 'Flight': '10', 'Carier': 'A'},
    {'Origin': 'EWR', 'station_id': '4', 'FL_DATETIME_UTC': datetime(2022, 3, 19, 3, 30), 'Flight': '11', 'Carier': 'B'}
]

# COMMAND ----------

# DBTITLE 1,Create a spark data frame from the test Airlines dataset
df_test_airlines_data = spark.createDataFrame(test_airlines_data)

# COMMAND ----------

display(df_test_airlines_data)

# COMMAND ----------

# DBTITLE 1,Create a test weather dataset
test_weather_data = [
    {'STATION': '1', 'DATE': datetime(2022, 3, 19, 1, 0), 'Temp': 15, 'Humidity': 5},
    {'STATION': '1', 'DATE': datetime(2022, 3, 19, 1, 50), 'Temp': 14, 'Humidity': 5},
    {'STATION': '1', 'DATE': datetime(2022, 3, 19, 2, 0), 'Temp': 16, 'Humidity': 3},
    {'STATION': '1', 'DATE': datetime(2022, 3, 19, 2, 50), 'Temp': 17, 'Humidity': 3},
    {'STATION': '1', 'DATE': datetime(2022, 3, 19, 3, 0), 'Temp': 18, 'Humidity': 2},
    {'STATION': '1', 'DATE': datetime(2022, 3, 19, 3, 50), 'Temp': 15, 'Humidity': 2},
    {'STATION': '1', 'DATE': datetime(2022, 3, 19, 4, 0), 'Temp': 16, 'Humidity': 1},
    {'STATION': '1', 'DATE': datetime(2022, 3, 19, 4, 50), 'Temp': 17, 'Humidity': 2},
    {'STATION': '1', 'DATE': datetime(2022, 3, 19, 5, 0), 'Temp': 18, 'Humidity': 6},
    {'STATION': '1', 'DATE': datetime(2022, 3, 19, 5, 50), 'Temp': 19, 'Humidity': 3},
    {'STATION': '1', 'DATE': datetime(2022, 3, 19, 6, 0), 'Temp': 15, 'Humidity': 2},
    {'STATION': '1', 'DATE': datetime(2022, 3, 19, 6, 50), 'Temp': 16, 'Humidity': 1},
    {'STATION': '1', 'DATE': datetime(2022, 3, 19, 7, 0), 'Temp': 18, 'Humidity': 1},
    {'STATION': '2', 'DATE': datetime(2022, 3, 19, 1, 0), 'Temp': 25, 'Humidity': 2},
    {'STATION': '2', 'DATE': datetime(2022, 3, 19, 1, 50), 'Temp': 26, 'Humidity': 3},
    {'STATION': '2', 'DATE': datetime(2022, 3, 19, 2, 0), 'Temp': 23, 'Humidity': 4},
    {'STATION': '2', 'DATE': datetime(2022, 3, 19, 2, 50), 'Temp': 25, 'Humidity': 5},
    {'STATION': '2', 'DATE': datetime(2022, 3, 19, 3, 0), 'Temp': 26, 'Humidity': 6},
    {'STATION': '5', 'DATE': datetime(2022, 3, 19, 1, 0), 'Temp': 30, 'Humidity': 2},
    {'STATION': '5', 'DATE': datetime(2022, 3, 19, 1, 50), 'Temp': 32, 'Humidity': 1},
    {'STATION': '5', 'DATE': datetime(2022, 3, 19, 1, 55), 'Temp': 29, 'Humidity': 1},    
]

# COMMAND ----------

# DBTITLE 1,Create a spark data frame from the test weather dataset
df_test_weather_data = spark.createDataFrame(test_weather_data)

# COMMAND ----------

display(df_test_weather_data)

# COMMAND ----------

# MAGIC %md Join the airlines and weather data on the weather station and two hours prior to departure time with a weather window of two.

# COMMAND ----------

from datetime import timedelta

def join_airlines_weather(df_airlines, df_weather, hrs_prior_to_departure = 2,  weather_window = 2):
    weather_window_start = (hrs_prior_to_departure + weather_window)*3600
    weather_window_end = hrs_prior_to_departure*3600
    df_airlines_weather = df_airlines.join(df_weather, 
                           (df_airlines.station_id == df_weather.STATION) & 
                           ((weather_window_end <= unix_timestamp(df_airlines.FL_DATETIME_UTC) - (unix_timestamp(df_weather.DATE))) & (weather_window_start >= unix_timestamp(df_airlines.FL_DATETIME_UTC) - unix_timestamp(df_weather.DATE))),
                          "left"
                          )
    return df_airlines_weather

# COMMAND ----------

df_airlines_weather_test = join_airlines_weather(df_test_airlines_data, df_test_weather_data)

# COMMAND ----------

display(df_airlines_weather_test)

# COMMAND ----------

# DBTITLE 1,Load Airlines Station Data
df_airlines_station_3m = spark.read.parquet(f"{blob_url}/df_airlines_station_3m")
df_airlines_station_6m = spark.read.parquet(f"{blob_url}/df_airlines_station_6m")

# COMMAND ----------

# DBTITLE 1,Join 3 month airlines data with the 3 months weather data.
df_airlines_weather_3m = join_airlines_weather(df_airlines_station_3m, df_weather_3m)

# COMMAND ----------

display(df_airlines_weather_3m)

# COMMAND ----------

# DBTITLE 1,Save the 3 months airlines_weather data
df_airlines_weather_3m.write.mode('overwrite').parquet(f"{blob_url}/df_airlines_weather_3m")

# COMMAND ----------

# DBTITLE 1,Join the 6 months airlines data with the 6 months weather data
df_airlines_weather_6m = join_airlines_weather(df_airlines_station_6m, df_weather_6m)

# COMMAND ----------

display(df_airlines_weather_6m)

# COMMAND ----------

# DBTITLE 1,Save the 6 months weather and airlines data
df_airlines_weather_6m.write.mode('overwrite').parquet(f"{blob_url}/df_airlines_weather_6m")

# COMMAND ----------

# MAGIC %md ### Run tests on the joins

# COMMAND ----------

# MAGIC %md Since there are two weather readings for every hour, for a two hour window, there should be around 4 weather readins except near the midnights when weather readings are less frequent.

# COMMAND ----------

# DBTITLE 1,Check if there are airlines with less than 4 weather readings
display(df_airlines_weather_6m.groupBy("FL_DATE_UTC", "DEP_HOUR_UTC", "OP_CARRIER_AIRLINE_ID", "OP_CARRIER_FL_NUM", "ORIGIN").count().where(col('count') < 4))

# COMMAND ----------

# DBTITLE 1,Check for a flight departing at midnight
display(df_airlines_weather_6m.where( (col('FL_DATE_UTC') == '2015-05-17') & 
                                     (col('ORIGIN') == 'ATL') & 
                                     (col('DEP_HOUR_UTC') == 0) & 
                                     (col('OP_CARRIER_FL_NUM') == '4738')))

# COMMAND ----------

# DBTITLE 1,Check for a flight departing at 1 AM
display(df_airlines_weather_6m.where( (col('FL_DATE_UTC') == '2015-05-15') & 
                                     (col('ORIGIN') == 'ATL') & 
                                     (col('DEP_HOUR_UTC') == 1) & 
                                     (col('OP_CARRIER_FL_NUM') == '1652')))

# COMMAND ----------

# DBTITLE 1,Check for a flight departing at 2 AM
display(df_airlines_weather_6m.where( (col('FL_DATE_UTC') == '2015-01-25') & 
                                     (col('ORIGIN') == 'ATL') & 
                                     (col('DEP_HOUR_UTC') == 2) & 
                                     (col('OP_CARRIER_FL_NUM') == '5610')))

# COMMAND ----------

display(df_airlines_weather_6m.where( (col('FL_DATE_UTC') == '2015-06-23') & 
                                     (col('ORIGIN') == 'ATL') & 
                                     (col('DEP_HOUR_UTC') == 1) & 
                                     (col('OP_CARRIER_FL_NUM') == '1833')))

# COMMAND ----------

# MAGIC %md check if there are any airlines missing weather data.

# COMMAND ----------

# DBTITLE 1,Check if station is null implying weather data is unavailable
df_missing_weather = df_airlines_weather_6m.where(col("STATION").isNull())

# COMMAND ----------

df_missing_weather.count()

# COMMAND ----------

df_airlines_weather_6m.count()

# COMMAND ----------

# DBTITLE 1,Check if weather readings are not after the flight departure
display(df_airlines_weather_3m.where((df_airlines_weather_3m.FL_DATE_UTC == '2015-03-04') & (df_airlines_weather_3m.DEP_HOUR_UTC == 20) & (df_airlines_weather_3m.OP_CARRIER_FL_NUM == '44')))