# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Predicting Flight Delays
# MAGIC 
# MAGIC ## Team 08 (section 5) -- W261 Final Project
# MAGIC 
# MAGIC ### Mrinal Chawla, Kumar Narayanan, Sushant Joshi, Ajeya Tarikere Jayaram
# MAGIC 
# MAGIC ### Instructor: Ramakrishna Gummadi

# COMMAND ----------

# MAGIC %md
# MAGIC # Abstract
# MAGIC 
# MAGIC The airline industry loses $40 billion anually due to delays [1]. This also leads to customer frustration with travel planning. Hence, there is a lucrative business opportunity for a model that can predict flight delays up 2 hours prior to departure. The research in this report uses weather readings and airport conditions in order to predict a departure delay of 15 minutes or more. While many different model configurations were tested, it was found that the ideal method involved a Random Forest model, utilizing 10 features, training on weather and airport data from 2015 to 2018 and tested on data from 2019. It showed that the traffic at airports was a big indicator of departure delay, while weather was only an indicator if certain key features, such as visibility, were affected. The final model had an accuracy of 84.4%. Detailed results are included below. This is a strong performance that can help improve customer experience. 
# MAGIC 
# MAGIC For future improvements, the model relied heavily on incoming delays to the origin airport and had poor results in cases where incoming delays did not correlate with outgoing delay, or when data about airport traffic was not available. Furthermore, the model struggled in particular with short distance flights. Additional feature engineering can help rectify these issues in future research. 

# COMMAND ----------

# MAGIC %md
# MAGIC # Introduction
# MAGIC 
# MAGIC ## Business Use Case
# MAGIC 
# MAGIC Predicting flight delays is a lucrative opportunity for airports and airlines alike. For a flight passenger, knowing the exact time of departure can help in planning transport to and from airports for pick up and drop off, itineraries for travel and provide ease of mind in a already stressful situation. If an airport or airline can consistently provide delay information to passengers, they are likely to see customer trust develop and a corresponding increase in revenue. Flight delays have a ripple effect and can cause disruptions to air transportation schedules. Delayed flights have a negative economic impact on the airlines operator, the air travelers, and on the airports. In the US, the impact of domestic flight delays was estimated to be $40 billion annually [1]. By being able to predict flights, air traffic control and airlines could put contingency measures in place to prevent this ripple effect and minimize the economic impact of flight delays.
# MAGIC 
# MAGIC ## Research Question
# MAGIC 
# MAGIC This problem is a classification problem, containing 2 possible classes, delayed or not delayed. Delayed will be defined as departure being 15 or more minutes after scheduled departure time. This determination needs to be made 2 hours before the scheduled departure, in order to provide benefits to the customers as discussed above. Concisely, the question can be formulated as: 
# MAGIC 
# MAGIC | _Will a flight's departure be delayed by 15 or more minutes, based on weather and airport conditions 2 hours prior to scheduled departure?_ |
# MAGIC |---|
# MAGIC 
# MAGIC ## Evaluation
# MAGIC 
# MAGIC To evaluate algorithms against this research question, the positive and negative classification classes must be defined. For this quesiton, a positive class will indicate that a flight was delayed by 15 or more minutes. A negative class will indicate that the flight was not delayed by 15 or more minutes. Extending from this definition, a false positive \\( (FP) \\) will indicate when an algorithm predicts a flight will be delayed but it is on time. A false negative \\( (FN) \\) will indicate when an algorithm predicts a flight will be on time, but it is delayed. A true positive \\( (TP) \\) and true negative \\( (TN) \\) will indicate when an algorithm predicts a delay or no delay correctly. From a customer's point of view, a false postive may lead to a missed flight, as the customer expects the flight to leave later than it does. A false negative will lead to extra stress as the customer expects the flight to leave earlier than it does. However, a false negative can be considered equivalent to the current status quo, as a customer would assume a flight would leave at its scheduled departure time without any information about predicted delays. 
# MAGIC 
# MAGIC With the parameters defined, there are several ways to measure performance of different predictive models. The most straightforward way would be to measure the accuracy, mathemetically defined as: \\(  \frac{TP + TN}{TP + FP + TN + FN} \\). However, the issue with this metric is the large skew in the dataset. There are many more flights that are not delayed than delayed, so an algorithm that only predicts not delayed could easily have a high accuracy score. Therefore, this paper will measure using the receiver operatator charatersitic curve (ROC). This curve plots the False Positive Rate \\( (FPR = \frac{FP}{FP+TN}) \\), equivalent to the precision, against the True Positive Rate \\( (TPR = \frac{TP}{TP+FN}) \\), equiavalent to the recall [8]. This gives a curve of how the model performs at all possible threshold values, from a probability threshold of 0, assuming all flights are delayed, to a threshold of 1, assuming all flights are not delayed. In order to compare models with a single metric, the Area Under the Curve (AUC) will also be reported. This allows for more insight into how many accurate positives are being generated, which would be hidden when using just the accuracy measure. 

# COMMAND ----------

# MAGIC %md
# MAGIC # Exploratory Data Analysis

# COMMAND ----------

#All imports
spark.sparkContext.addPyFile("dbfs:/custom_cv.py")

import matplotlib.pyplot as plt
import pandas as pd
import random
import seaborn as sns

from custom_cv import CustomCrossValidator
from functools import reduce
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, VectorSlicer, StringIndexer, OneHotEncoder
from pyspark.ml.stat import Correlation
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.mllib.evaluation import MulticlassMetrics, BinaryClassificationMetrics
from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import col, floor, when, desc, count, max, avg, first, min, mean, struct, lit, row_number, sum
from pyspark.sql.types import FloatType, IntegerType
from pyspark.sql.functions import col

# COMMAND ----------

#Paths for data loading
blob_container = "team08-container" # The name of your container created in https://portal.azure.com
storage_account = "team08" # The name of your Storage account created in https://portal.azure.com
secret_scope = "team08-scope" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "team08-key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Dataset
# MAGIC 
# MAGIC ### Representation 
# MAGIC 
# MAGIC With a large dataset, the data needs to be represented efficiently in the cluster. It was found that the best representation would be a column based representation, such as parquet. Column based storage stores all the values for a particular column close together in memory. This allows queries with filters to run much more efficiently, as only a few bytes for the column have to be read, instead of the entire row, as with a row based storage solution. Furthermore, if the filtering can be processed with only a few bytes, the entire row can be read post filtering, which saves much more time in reading. This is an ideal solution for this type of data where there are many columns in each data set, particularily once the join is done. 
# MAGIC 
# MAGIC The three datasets employed to assist with the task of predicting flight delays are:
# MAGIC 
# MAGIC   1. **Flights dataset**, a subset of the passenger flight's on-time performance data between 2015 and 2019 taken from the [TranStats data](https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236&DB_Short_Name=On-Time) collection available from the U.S. Department of Transportation (DOT). This dataset contains flight information for 31,746,841 flights and has 109 attributes.  There are 5,693,541 flights that are delayed, 25,576,004 flights that were not delayed and 477,296 flights with other status.
# MAGIC   1. **Weather Dataset** from the [National Oceanic and Atmospheric Administration repository](https://www.ncdc.noaa.gov/orders/qclcd/) for the period between 2015 and 2019. This dataset contains weather information collected from weather stations across the United States and other parts of the world. The weather dataset contains 630,904,436 weather readings and has 177 different weather attributes.
# MAGIC 
# MAGIC   1. **Station Dataset** This dataset contains geolocation information of 2,237 weather stations located in the United States of America and the distance from these weather stations to other weather stations. There are 5,004,169 entries with 12 attributes. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## [Weather Dataset](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1898361324233996/command/1898361324233997)
# MAGIC 
# MAGIC | Description | Notes |
# MAGIC |---|---|
# MAGIC | Number of Datapoints in full Data Set |  630,904,436 | 
# MAGIC | Number of Datapoints in 1st Quarter of 2015 | 29,823,926 |
# MAGIC | Total Number of Features in Raw | 177 |
# MAGIC | Total Number of Retained Features | 13 |
# MAGIC | Interesting Feature Groups | Air Temperature, Dew Point Temperature, Windspeed, Visibility, Sea level Pressure, Precipitation, Snow, Fog, Thunder, Gust among other available parameters |
# MAGIC | Required Features | Date, Latitude, Longitude, Station |
# MAGIC | Dropped Features | Quality Code Features, Features from Additional Section |

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The [weather data sheet](https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf) is divided into three groups - Control Section, Mandatory Data Section and Additional Data Sheet. The Control Section consists of the weather station information while the mandatory data section includes temperature, windspeed, horizontal and vertical visibility. The additional data section includes other weather parameters such as precipitation, snow, extreme weather conditions, past and present weather conditions. The weather features that are available in the mandatory data section are dense and of higher data quality, while the additional data sections are quite sparse and have large percentages of missing values (>90% in many cases). The steps that were performed are discussed in detail in [exploratory data analysis](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1898361324233996/command/1898361324233997) phase for the weather data. 
# MAGIC 
# MAGIC Based on the analysis, the weather parameters that were retained are as follows: 
# MAGIC `Mandatory Data Section: WIND_DIR, WIND_SPEED_RATE, WIND_TYPE, VIS_DISTANCE, CEILING_HEIGHT, AIR_TEMP, DEW_POINT_TEMPERATURE, SEA_LEVEL_PRESSURE`
# MAGIC `Additional Data Section: ATMOS_PRESSURE_3H, PRECIPITATION, PRESENT_ACC, PAST_ACC, ATMOS_PRESSURE_TENDENCY_3H`
# MAGIC 
# MAGIC The correlation matrix below captures some of the correlations among some of the mandatory weather features.

# COMMAND ----------

df_weather = spark.read.parquet(f"{blob_url}/df_weather")

# convert to vector column first
vector_col = "corr_features"
inputColumns = ["WIND_DIR_IMP", "WIND_SPEED_RATE_IMP", "CEILING_HEIGHT_IMP", "VIS_DISTANCE_IMP", "AIR_TEMP_IMP", "DEW_POINT_TEMPERATURE_IMP", "SEA_LEVEL_PRESSURE_IMP"]
assembler = VectorAssembler(inputCols=inputColumns, outputCol=vector_col)
df_vector = assembler.transform(df_weather).select(vector_col)

# get correlation matrix
matrix = Correlation.corr(df_vector, vector_col).collect()[0][0]

corrmatrix = matrix.toArray().tolist()
cmap = sns.light_palette("#4e909e", as_cmap=True)
plt.title("Correlation Matrix", fontsize =20)
sns.heatmap(corrmatrix, 
        xticklabels=inputColumns,
        yticklabels=inputColumns, cmap=cmap, annot=True, vmin=-1, vmax=1)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC From the correlation matrix, it can be observed that there is correlation betwen air and dew point temperature as well as a correlation between visibility distance and ceiling height. There also exists a negative correlation between sea level pressure and both the temperature parameters. 
# MAGIC 
# MAGIC The weather features have had a range of data availability and data quality issues that were important to be taken into account, specially when handling the case of null and missing values. This is especially true when we need to impute the values for a feature where we have very sparse data. An additional fact that had to be accounted for is that a missing value could actually be a missing value or represented by the station as 99 or 999. Different weather features have different set of indicators for missing values as well. This is further compounded by a co-paired column in the dataframe which is an indicator of the data quality of the given feature which has its own categorical encoding. All these factors have been taken into account for the features that were selected.
# MAGIC 
# MAGIC To illustrate with an example, it is uncommon to observe heavy rain / precipitation over large periods of time in continuum. The airlines-weather data join is discussed in the subsequent sections below, but the key point is that imputation had to be applied over the consistent window length for given the weather stations recording, such that the effect of the aggregation function does not over or under emphasize the impact of that weather feature. [Here](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1038316885895986/command/1898361324243719) is the code snippet that demonstrates how this imputation was achieved and applied.
# MAGIC 
# MAGIC There are a set of weather features which are categorical features binned into distinct numerical values. One Hot Encoding has been applied for such variables, as applicable.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [Flights Dataset](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1898361324231581/command/1898361324231584)
# MAGIC | Description | Notes |
# MAGIC |---|---|
# MAGIC | Total Number of DataPoints in Full Data Set | 31,746,841 |
# MAGIC | Total Number of DataPoints in 1st Quarter of 2015 | 161,057 |
# MAGIC | Total Number of Features in original data set | 109 |
# MAGIC | Number of Categorical Features | 22 |
# MAGIC | Number of Numerical Features | 85 |
# MAGIC | Total Number of Features with >70% missing values | 50 |
# MAGIC | Interesting Feature Groups | Time Period, Airline, Origin and Destination, Departure and Arrival Performance, Flight Summaries, Delay Causes, Diverted Airport Information |
# MAGIC 
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
# MAGIC |Diverted Airport Information| DIV_AIRPORT_LANDINGS, DIV_REACHED_DEST, DIV_ACTUAL_ELAPSED_TIME, DIV_ARR_DELAY, DIV_DISTANCE, DIV1_AIRPORT, DIV1_AIRPORT_ID, DIV1_AIRPORT_SEQ_ID, DIV1_WHEELS_ON, DIV1_TOTAL_GTIME, DIV1_LONGEST_GTIME, DIV1_WHEELS_OFF, DIV1_TAIL_NUM, DIV2_AIRPORT, DIV2_AIRPORT_ID, DIV2_AIRPORT_SEQ_ID, DIV2_WHEELS_ON, DIV2_TOTAL_GTIME, DIV2_LONGEST_GTIME, DIV2_WHEELS_OFF, DIV2_TAIL_NUM, DIV3_AIRPORT, DIV3_AIRPORT_ID, DIV3_AIRPORT_SEQ_ID, DIV3_WHEELS_ON, DIV3_TOTAL_GTIME, DIV3_LONGEST_GTIME, DIV3_WHEELS_OFF, DIV3_TAIL_NUM, DIV4_AIRPORT, DIV4_AIRPORT_ID, DIV4_AIRPORT_SEQ_ID, DIV4_WHEELS_ON, DIV4_TOTAL_GTIME, DIV4_LONGEST_GTIME, DIV4_WHEELS_OFF, DIV4_TAIL_NUM, DIV5_AIRPORT, DIV5_AIRPORT_ID, DIV5_AIRPORT_SEQ_ID, DIV5_WHEELS_ON, DIV5_TOTAL_GTIME, DIV5_LONGEST_GTIME, DIV5_WHEELS_OFF, DIV5_TAIL_NUM|

# COMMAND ----------

#Load raw airlines data from mount
df_airlines = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data/20*")

# COMMAND ----------

# MAGIC %md The airlines data is heavily skewed by the flight departure status. Delayed flights are around one-fourth of the flights that departed on-schedule.

# COMMAND ----------

#display skew of output variable
display(df_airlines
        .withColumn("Departure Status", 
                    when(col("Dep_Del15") == "0", "On-time")
                    .when(col("Dep_Del15") == "1", "Delayed")
                    .otherwise("Other"))
        .select("Departure Status")
        .groupBy("Departure Status")
        .count())

# COMMAND ----------

# MAGIC %md When the flight status by departure hour is plotted, it can be observed that more flights seem to be delayed in the later hours of the day, peaking between 4 PM and 8 PM.

# COMMAND ----------

#Display delay by departure hour
display(df_airlines
        .withColumn('Hour', floor(col('CRS_Dep_Time')/100))
        .withColumn("Departure Status", 
                    when(col("Dep_Del15") == "0", "On-time")
                    .when(col("Dep_Del15") == "1", "Delayed")
                    .otherwise("Other"))
        .select("Hour", col("Departure Status"))
        .groupBy("Departure Status", "Hour")
        .count()
        .sort("hour", "Departure Status"))

# COMMAND ----------

# MAGIC %md The plot of number of flights by an operator/carrier reveals that operators/carriers with more flights also tends to have more delays. Airlines such as Southwest Airlines (WN), Delta Airlines (DL), American Airlines (AA), Skywest Airlines (OO), United Airlines (UA) have more flights and also more departure delays. 

# COMMAND ----------

#Delay by carrier 
display(df_airlines
        .withColumn("Departure Status", 
                    when(col("Dep_Del15") == "0", "On-time")
                    .when(col("Dep_Del15") == "1", "Delayed")
                    .otherwise("Other"))
        .select("Op_Carrier", "Departure Status")
        .groupBy("Op_Carrier", "Departure Status")
        .count()
        .sort(desc("count"))
       .limit(20))

# COMMAND ----------

# MAGIC %md A similar plot of the flight departure status at the origin airports shows that airports that have more flights also generally have more delays.

# COMMAND ----------

#Delays by origin
df_origin_delays = df_airlines \
        .select("Origin") \
        .where(col("Dep_Del15") == "1")\
        .groupBy("Origin")\
        .count()

#Non-delay by origin
df_origin_no_delays = df_airlines\
        .select("Origin")\
        .where(col("Dep_Del15") == "0")\
        .groupBy("Origin")\
        .count()

#Combine delays / non delays by origin, rename, and plot
df_origin = df_origin_delays.withColumnRenamed("count", "delays").join(df_origin_no_delays.withColumnRenamed("count", "on-time"), "Origin")
        
display(df_origin
        .withColumn("total", col('delays') + col('on-time'))
        .sort(desc("total"))
        .limit(20))

# COMMAND ----------

# MAGIC %md Plotting departure delays by reason for delays shows that late aircrafts, carriers, and the National Airspace System (NAS) are the major reasons for delays. Though weather makes up a small fraction of delays in the plot, the disruptive effects of delays caused by massive weather conditions on air traffic network cannot be discounted.

# COMMAND ----------

#Select delays by reason and rename for labels
df_delay_reasons_carrier = df_airlines.where((col("CARRIER_DELAY") > 0) & (col("DEP_DEL15") == 1)).select(count("CARRIER_DELAY").alias("count")).withColumn("Reason", lit("Carrier"))
df_delay_reasons_weather = df_airlines.where((col("WEATHER_DELAY") > 0) & (col("DEP_DEL15") == 1)).select(count("WEATHER_DELAY").alias("count")).withColumn("Reason", lit("Weather"))
df_delay_reasons_nas = df_airlines.where((col("NAS_DELAY") > 0)  & (col("DEP_DEL15") == 1)).select(count("NAS_DELAY").alias("count")).withColumn("Reason", lit("Nas"))
df_delay_reasons_security = df_airlines.where((col("SECURITY_DELAY") > 0)  & (col("DEP_DEL15") == 1)).select(count("SECURITY_DELAY").alias("count")).withColumn("Reason", lit("Security"))
df_delay_reasons_aircraft = df_airlines.where((col("LATE_AIRCRAFT_DELAY") > 0) & (col("DEP_DEL15") == 1)).select(count("LATE_AIRCRAFT_DELAY").alias("count")).withColumn("Reason", lit("Late Aircraft"))

#Plot delays by reason
df_delay_reasons = reduce(DataFrame.unionAll, [df_delay_reasons_carrier, df_delay_reasons_weather, df_delay_reasons_nas, df_delay_reasons_security, df_delay_reasons_aircraft])
display(df_delay_reasons)

# COMMAND ----------

# MAGIC %md Plotting the correlation matrix of the temporal features and distance reveals that there is strong correlationship between month and quarter.

# COMMAND ----------

# convert to vector column first
vector_col = "corr_features"
inputColumns = ["YEAR", "QUARTER", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "DISTANCE"]
assembler = VectorAssembler(inputCols=inputColumns, outputCol=vector_col)
df_vector = assembler.transform(df_airlines).select(vector_col)

# get correlation matrix
matrix = Correlation.corr(df_vector, vector_col).collect()[0][0]
corrmatrix = matrix.toArray().tolist()
cmap = sns.light_palette("#4e909e", as_cmap=True)
plt.title("Correlation Matrix", fontsize =20)
sns.heatmap(corrmatrix, 
        xticklabels=inputColumns,
        yticklabels=inputColumns, cmap=cmap, annot=True, vmin=-1, vmax=1)

# COMMAND ----------

# MAGIC %md
# MAGIC The plot of flight departure status and any prior delays at the airport in a 2 hour time window 2 hours prior to a flight's departure time reveals that whenever there are flight delays at an airport, there were relatively more delays at the airport in the 2 hour window 2 hrs prior to the departure time.

# COMMAND ----------

#Load cleaned airlines data
df_airlines_clean = spark.read.parquet(f"{blob_url}/airline_data_clean")

# COMMAND ----------

#Delays by prior delays at airport
display(df_airlines_clean
        .withColumn("Departure Status", 
                    when(col("Dep_Del15") == "0", "On-time")
                    .when(col("Dep_Del15") == "1", "Delayed")
                    .otherwise("Other"))
        .withColumn("Any Prior Delays at the airport?", when(col("number_dep_delay_origin") > 0, "Yes").otherwise("No"))
        .select("Departure Status", "Any Prior Delays at the airport?")
        .groupBy("Departure Status", "Any Prior Delays at the airport?")
        .count())

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC A plot of flight departure status and any prior arrival delays for an aircraft in a 2 hour time window 2 hours prior to a flight's departure time reveals that whenever there are arrival delays for an aircraft, there were relatively more arrival delays for the aircraft in the 2 hour window 2 hrs prior to the departure time.

# COMMAND ----------

#Delays by prior delays by aircraft
display(df_airlines_clean
        .withColumn("Departure Status", 
                    when(col("Dep_Del15") == "0", "On-time")
                    .when(col("Dep_Del15") == "1", "Delayed")
                    .otherwise("Other"))
        .withColumn("Any Prior Delays for Aircraft?", when(col("number_arr_delay_tail") > 0, "Yes").otherwise("No"))
        .select("Departure Status", "Any Prior Delays for Aircraft?")
        .groupBy("Departure Status", "Any Prior Delays for Aircraft?")
        .count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## [Stations Dataset](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1858507102399370/command/1858507102400296)
# MAGIC 
# MAGIC | Description | Notes |
# MAGIC |---|---|
# MAGIC | Number of Datapoints in full Data Set |  5,004,169 | 
# MAGIC | Total Number of Features in Raw | 12|
# MAGIC | Total Number of Retained Features | 5 |
# MAGIC | Required Features | usad, wban, station_id (the unique IDs of a station), lat (its latitude), lon (its longitude), distance_to_neighbor |
# MAGIC | Dropped Features | neighbor_id, neighbor_name, neighbor_state, neighbor_call, neighbor_lat, neighbor_lon |
# MAGIC 
# MAGIC * The station data set consists of the following important columns
# MAGIC   * "wban" (Weather Bureau Army Navy): which is a 5-digit number as one of the unique ways to identify weather station
# MAGIC   * usaf" (United States Air Force): another numbering scheme for weather stations
# MAGIC   * "station_id": This column is concatenation of usaf and wban (in that order) and provides an unique ID. Each station ID occurs 2237 times
# MAGIC     * Each station has information related to every other station, including itself
# MAGIC     * Each of the 2237 stations reports its name, other station's name, distance to other stations etc, for a total of 2237x2237 = 5,0004,169 entires
# MAGIC   * The columns "lat" and "lon" provides the latitude and longitude respectively.
# MAGIC   * Other columns are more descriptive in nature and isn't used in the analysis. The number of states are 52, and includes Puerto Rico and Virgin Islands.

# COMMAND ----------

# MAGIC %md
# MAGIC ![airport_station_join](https://team08.blob.core.windows.net/team08-container/airport_weatherStation_join.png)

# COMMAND ----------

#Load stations data
df_stations = spark.read.parquet("/mnt/mids-w261/datasets_final_project/stations_data/*")
display(df_stations)

# COMMAND ----------

print("Number of datapoints in the dataset:", df_stations.count())
print("Number of unique stations:", df_stations.select("station_id").distinct().count())

# COMMAND ----------

df1 = df_stations.select("neighbor_state").groupBy("neighbor_state").count()

display(df1.sort(col("count")))

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC ### [Joins](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1898361324236910/command/1898361324236913)
# MAGIC 
# MAGIC Inclement weather conditions could delay flight departures. Hence, knowing the weather conditions around the time of the flight would be helpful to predict flight delays. As the airline operators and the airport staff need sufficient time to take measures to address the flight delays, weather delays have to be predicted at least 2 hours prior to a flight's departure time.
# MAGIC 
# MAGIC In order to get the weather at the airport, the closest weather station to the airport had to be identified. However, though the flights data had the codes for the origin and destination airports, it had no location information about these airports. To get the geolocation of the airport, a python package geopy was employed to lookup the geocoordinates (latitude and longitude) for the airports. The code to look up airport geolocations is available [here](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/893260312290660/command/893260312290661).
# MAGIC 
# MAGIC To get the [closest weather station to an airport](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/3010530795775164/command/3816960191371678), **stations which have themselves as the neighbors were filtered** as the stations table contains the distance of a station to all neighboring stations along with distance to itself. An airport_stations_lookup table was created by **computing the haversine distance** from an airport to a weather station in the stations table by cross joining the airport codes and unique stations, and filtering only the stations which had the smallest distance to the airport. Next, a left join was applied on airlines table and the airport_stations_lookup table on the **airport code** of the origin airport.
# MAGIC 
# MAGIC The flights table had departure time in local or origin airport timezone, whereas the weather table had date in UTC, so the flight departure time was converted to UTC to facilitate the join on weather table. Next a [left join was applied on the weather table](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1898361324236910/command/1898361324236913) on the **station** field where the **date** timestamp is between 2 hours prior to the **departure time of the flight** and 4 hours prior to the departure time. This weather window of 2 hours was used to get a better estimate of weather conditions at the airport. For example, if the flight was scheduled to depart at 10 AM, the weather reading availables between 6 AM and 8 AM on that day were used to determine the average weather condition. A pictorial represention of the join is given below.

# COMMAND ----------

# MAGIC %md
# MAGIC ![my_test_image](https://team08.blob.core.windows.net/team08-container/DataJoin_workflow.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Handling missing values
# MAGIC 
# MAGIC From the flights table, cancelled flights were dropped since they the task is to make predictions about flight delays. 
# MAGIC 
# MAGIC In the weather table, columns 'WND', 'CIG', 'VIS', 'TMP', 'DEW', 'SLP', 'MW1', 'AY1', 'MD1', 'OC1', 'AA1' had weather readings that were combined together into comma separated strings. The individual measurements were extracted by splitting these strings and indexing the appropriate measurements. The method is included in the cell below. 

# COMMAND ----------

def split_columns(df):
    """
    Method to split up weather columns. Original data has multiple readings in single column
    Input: Raw dataframe
    Output: Dataframe with columns split
    """
    
    #Split columns that have multiple readings
    wind_split_col = split(df['WND'], ',')
    cig_split_col = split(df['CIG'], ',')
    vis_split_col = split(df['VIS'], ',')
    tmp_split_col = split(df['TMP'], ',')
    dew_split_col = split(df['DEW'], ',')
    slp_split_col = split(df['SLP'], ',')

    #Additional Feature splits
    mw1_split_col = split(df['MW1'], ',')
    ay1_split_col = split(df['AY1'], ',')
    md1_split_col = split(df['MD1'], ',')
    oc1_split_col = split(df['OC1'], ',')
    aa1_split_col = split(df['AA1'], ',')

    #Add individual features as individual columns
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

df_weather = split_columns(df_weather)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Many of these features used the highest value in the range of values the feature can have such as 999 for wind direction (WIND_DIR), 9999 for wind speed rate (WIND_SPEED_RATE), 999999 for visual distance (VIS_DISTANCE) etc to indicate a missing value. So, for such features, if the feature had these highest values or the feature was null, the average (mean) value of the last 720 weather readings (approximately 15 days of weather readings, as most stations have 2 weather readings per hour) at the station were imputed. The code snippet to impute the average value for these missing values is shown below and the full code is [available in this notebook](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1898361324233996/command/1898361324244302).
# MAGIC 
# MAGIC The full table of missing values is given below:
# MAGIC 
# MAGIC |Base Feature|Missing Value Indicator|
# MAGIC |---|---|
# MAGIC ELEVATION | 9999|
# MAGIC WIND_DIR | 999|
# MAGIC WIND_TYPE | 9 |
# MAGIC WIND_SPEED_RATE | 9999 |
# MAGIC CEILING_HEIGHT | 99999 |
# MAGIC VIS_DISTANCE | 999999 |
# MAGIC AIR_TEMP | 9999 |
# MAGIC DEW_POINT_TEMPERATURE | 9999 |
# MAGIC SEA_LEVEL_PRESSURE | 99999 |
# MAGIC PAST_ATMOSPHERIC_CONDITION_CODE_DURATION | 99 |
# MAGIC ATMOS_PRESSURE_TENDENCY_3H | 9 |
# MAGIC ATMOS_PRESSURE_3H | 999 |
# MAGIC WIND_GUST_SPEED_RATE | 9999 |
# MAGIC PRECIPITATION_PERIOD_QTY | 99 |
# MAGIC PRECIPITATION_DEPTH | 9999 |

# COMMAND ----------

#Map of feature to corresponding missing code
feature_mcode_mapping = {"WIND_DIR":999, "WIND_SPEED_RATE":9999, "VIS_DISTANCE":999999,"CEILING_HEIGHT":99999, "AIR_TEMP":9999, "DEW_POINT_TEMPERATURE":9999, "SEA_LEVEL_PRESSURE":99999, "ATMOS_PRESSURE_3H":999 }

def replace(column, value):
    '''This function replaces missing values with null'''
    return when(column != value, column).otherwise(lit(None))

#Replace all missing values for features
for feature, missing_code in feature_mcode_mapping.items():
    df_weather = df_weather.withColumn(feature, replace(col(feature), missing_code))

def apply_avg_imputation(df, col_name, missing_code):
    '''This function computes the average(mean) of the last 720 records at the station for a given feature'''
    window = Window.partitionBy("STATION").orderBy("DATE").rowsBetween(-720, 0)
    condition = when( (col(col_name).isNull()) | (col(col_name) == missing_code), floor(mean(col(col_name)).over(window))).otherwise(col(col_name))
    return df.withColumn(col_name+"_IMP", condition)     

#Impute in missing values
for column, missing_code in feature_mcode_mapping.items():
    df_weather=apply_avg_imputation(df_weather, column, missing_code)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC For PRECIPITATION_PERIOD_QTY, WIND_GUST_SPEED_RATE, WIND_TYPE, PRESENT_ATMOSPHERIC_CONDITION_CODE, PAST_ATMOSPHERIC_CONDITION_CODE, ATMOS_PRESSURE_TENDENCY_3H empty values were replaced with NA or the corresponing missing value code for the feature.

# COMMAND ----------

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
# MAGIC Many of the columns that had too many missing values (nearly 95% values missing) or were already extracted into other features or feature that were excluded were dropped from the table. The list of columns dropped are given below:
# MAGIC 
# MAGIC |Columns Dropped|
# MAGIC |-|
# MAGIC |"AW1","GA1","GA2","GA3","GA4","GE1","GF1","KA2","MA1","OD1","OD2","REM","EQD","AW2","AX4","GD1",
# MAGIC "AW5","GN1","AJ1","AW3","MK1","KA4","GG3","AN1","RH1","AU5","HL1","OB1","AT8",
# MAGIC "AW7","AZ1","CH1","RH3","GK1","IB1","AX1","CT1","AK1","CN2","OE1","MW5","AO1",
# MAGIC "KA3","AA3","CR1","CF2","KB2","GM1","AT5","MW6","MG1","AH6","AU2","GD2","AW4",
# MAGIC "MF1","AH2","AH3","OE3","AT6","AL2","AL3","AX5","IB2","AI3","CV3","WA1","GH1",
# MAGIC "KF1","CU2","CT3","SA1","AU1","KD2","AI5","GO1","GD3","CG3","AI1","AL1","AW6",
# MAGIC "MW4","AX6","CV1","ME1","KC2","CN1","UA1","GD5","UG2","AT3","AT4","GJ1","MV1",
# MAGIC "GA5","CT2","CG2","ED1","AE1","CO1","KE1","KB1","AI4","MW3","KG2","AA2","AX2",
# MAGIC "RH2","OE2","CU3","MH1","AM1","AU4","GA6","KG1","AU3","AT7","KD1","GL1","IA1",
# MAGIC "GG2","OD3","UG1","CB1","AI6","CI1","CV2","AZ2","AD1","AH1","WD1","AA4","KC1",
# MAGIC "IA2","CF3","AI2","AT1","GD4","AX3","AH4","KB3","CU1","CN4","AT2","CG1","CF1",
# MAGIC "GG1","MV2","CW1","GG4","AB1","AH5","CN3","AY2","KA1", "SLP","DEW","TMP","VIS",
# MAGIC "CIG","WND","MD1","MW1","MW2","OC1","AA1","AY1",
# MAGIC "LATITUDE","LONGITUDE","SOURCE","REPORT_TYPE","QUALITY_CONTROL","NAME","CALL_SIGN",
# MAGIC "AIR_TEMP_QUALITY_CODE","DEW_POINT_QUALITY_CODE","SEA_LEVEL_PRESSURE_QUALITY_CODE",
# MAGIC "WIND_DIR_QUALITY_CODE","WIND_SPEED_QUALITY_CODE","VIS_DISTANCE_QUALITY_CODE","CEILING_QUALITY_CODE",
# MAGIC "CEILING_DETERM_CODE","CAVOK","VIS_DISTANCE_QUALITY_CODE","VIS_QUALITY_VARIABILITY_CODE","VIS_VARIABILTY_CODE"|

# COMMAND ----------

# MAGIC %md
# MAGIC ## New Features
# MAGIC 
# MAGIC Additional features that were computed from the flights table are given below:
# MAGIC 
# MAGIC 1. DEP_HOUR - since the EDA revealed that the departures in the evening tend to have more delays, the hour of departure was computed as a feature
# MAGIC 1. number_dep_delay_origin - the number of flights at the origin airport that had a departure delay in a 2 hour window 2 hours prior to the departure of a flight. 
# MAGIC 1. number_flights_origin - total number of flights departing from the origin airport in a 2 hour window 2 hours prior to the departure of a flight.
# MAGIC 1. percent_dep_delay_origin - the percentage of flights that were delayed at the origin airport in a 2 hour window 2 hours prior to the departure of a flight.
# MAGIC 1. number_arr_delay_tail - the number of times the aircraft had an arrival delay in a 2 hour window 2 hours prior to the departure of the flight.
# MAGIC 1. number_flights_tail - the number of flights the aircraft had in a 2 hour window 2 hours prior to the departure of the flight.
# MAGIC 1. percent_arr_delay_tail - the percentage of arrival delays that the aircraft had in a 2 hour window 2 hours prior to the departure of the flight.
# MAGIC 
# MAGIC The full code for these feature computations can be found [here](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1898361324231581/command/1898361324231584)

# COMMAND ----------

def compute_departure_delays(dataframe):
    '''Function to compute the departure delays at the origin airport in a 2 hour window 2 hours prior to the departure of a flight.'''
    
    #Prepare for SQL useage
    dataframe.createOrReplaceTempView("airlines")
    
    #Select raw numbers needed for feature computation
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
    
    #Add in new features
    df_updated = df_updated \
                        .withColumn("number_dep_delay_origin", col("number_dep_delay_origin_4h") - col("number_dep_delay_origin_2h")) \
                        .withColumn("number_flights_origin", col("number_flights_origin_4h") - col("number_flights_origin_2h")) \
                        .drop("number_dep_delay_origin_4h", "number_dep_delay_origin_2h", "number_flights_origin_4h", "number_flights_origin_2h")
    return df_updated

def compute_arrival_delays(dataframe):
    '''Function to compute the arrival delays for an aircraft in a 2 hour window 2 hours prior to it's departure.'''
    
    #Prepare for SQL
    dataframe.createOrReplaceTempView("airlines")
    
    #Select raw numbers
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
    
    #Create new features
    df_updated = df_updated \
                        .withColumn("number_arr_delay_tail", col("number_arr_delay_tail_4h") - col("number_arr_delay_tail_2h")) \
                        .withColumn("number_flights_tail", col("number_flights_tail_4h") - col("number_flights_tail_2h")) \
                        .drop("number_arr_delay_tail_4h", "number_arr_delay_tail_2h", "number_flights_tail_4h", "number_flights_tail_2h")
  
  return df_updated

df_airlines = compute_departure_delays(df_airlines)
df_airlines = compute_arrival_delays(df_airlines)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC In the weather table, additional features which give the overall average for each of the features - WIND_DIR, WIND_SPEED_RATE, VIS_DISTANCE, CEILING_HEIGHT, AIR_TEMP, DEW_POINT_TEMPERATURE, SEA_LEVEL_PRESSURE and ATMOS_PRESSURE_3H were created.

# COMMAND ----------

def apply_avg_imputation_across_feature(df, feature_name):
    """Function to compute the average value of the given feature name"""
    window = Window.partitionBy()
    condition = when(col(feature_name).isNull(), floor(mean(col(feature_name)).over(window))).otherwise(col(feature_name))
    return df.withColumn(feature_name, condition)      

for feature, missing_code in feature_mcode_mapping.items():
    df_weather=apply_avg_imputation_across_feature(df_weather, feature+"_IMP")

# COMMAND ----------

# MAGIC %md
# MAGIC A new feature called PRECIPITATION_RATE which is the PRECIPITATION_DEPTH / PRECIPITATION_PERIOD_QTY was created and the features PRECIPITATION_PERIOD_QTY and PRECIPITATION_DEPTH were dropped.

# COMMAND ----------

#Replace missing with 0
df_weather = df_weather.na.replace(99, 0, subset = ["PRECIPITATION_PERIOD_QTY"]) \
                       .na.replace(9999, 0, subset = ["PRECIPITATION_DEPTH"]) 

#Compute rate
df_weather = df_weather.withColumn("PRECIPITATION_RATE", df_weather["PRECIPITATION_DEPTH"] / df_weather["PRECIPITATION_PERIOD_QTY"]).na.fill(0, subset=["PRECIPITATION_RATE"]) 

#drop depth and period
df_weather = drop_columns(df_weather, ["PRECIPITATION_PERIOD_QTY", "PRECIPITATION_DEPTH"])

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Categorical columns such as PRESENT_ATMOSPHERIC_CONDITION_CODE, PAST_ATMOSPHERIC_CONDITION_CODE, ATMOS_PRESSURE_TENDENCY_3H and WIND_TYPE were transformed into one-hot encoded vectors and the categorical columns were dropped.

# COMMAND ----------

#Create indicies for the columns
indexer = StringIndexer(inputCols=["PRESENT_ATMOSPHERIC_CONDITION_CODE","PAST_ATMOSPHERIC_CONDITION_CODE","ATMOS_PRESSURE_TENDENCY_3H","WIND_TYPE"], outputCols=["PRESENT_ACC_INDEX","PAST_ACC_INDEX","ATMOS_PRESSURE_TENDENCY_3H_INDEX","WIND_TYPE_INDEX"])
df_weather = indexer.fit(df_weather).transform(df_weather)

#One hot encode columns by index
ohe = OneHotEncoder(inputCols=["PRESENT_ACC_INDEX","PAST_ACC_INDEX","ATMOS_PRESSURE_TENDENCY_3H_INDEX","WIND_TYPE_INDEX"], outputCols=["PRESENT_ACC_OHE","PAST_ACC_OHE","ATMOS_PRESSURE_TENDENCY_3H_OHE","WIND_TYPE_OHE"])
df_weather=ohe.fit(df_weather).transform(df_weather)

# Drop base columns since OHE columns available now, as well as the index columns since they are not needed
cols = {"PRESENT_ATMOSPHERIC_CONDITION_CODE","PAST_ATMOSPHERIC_CONDITION_CODE","ATMOS_PRESSURE_TENDENCY_3H","WIND_TYPE","PRESENT_ACC_INDEX","PAST_ACC_INDEX","ATMOS_PRESSURE_TENDENCY_3H_INDEX","WIND_TYPE_INDEX"}
df_weather=drop_columns(df_weather,cols)

# COMMAND ----------

# MAGIC %md The full code for the weather feature computations can be found [here](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1898361324233996/command/1898361324233997)

# COMMAND ----------

# MAGIC %md
# MAGIC # Algorithms

# COMMAND ----------

#Load all data
df_airlines_weather_full = spark.read.parquet(f"{blob_url}/df_airlines_weather_holidays")
df_airlines_weather_full.cache()
df_airlines_weather = df_airlines_weather_full.where(df_airlines_weather_full.FL_DATE_UTC < '2019-01-01')
df_airlines_weather.cache()
df_airlines_weather_19 = df_airlines_weather_full.where(df_airlines_weather_full.FL_DATE_UTC >= '2019-01-01')
df_airlines_weather_19.cache()

# COMMAND ----------

# DBTITLE 0,Untitled
# define a class that build on BinaryClassificationMetrics
class CurveMetrics(BinaryClassificationMetrics):
    """The input has to be in the form of a RDD with probability and label. As an example:
    predictions.select('label','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['label'])))
    """
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
    
    # methods are 'roc' and 'pr'
    def get_curve(self, method):
        rdd = getattr(self._java_model, method)().toJavaRDD()
        return self._to_list(rdd)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Baseline Model
# MAGIC 
# MAGIC In order to establish a baseline for improvement, the average rate of delayed flights is used. This model calculates the rate of delayed flights in the dataset using the formula below. Then it randomly generates a prediction based on that rate for each record in the test dataset. This is equivalent to a customer making an educated guess about delays. The included ROC curve is a linear line and AUC value if 0.499. These metrics are used as the minimum that a model should acheive to be considered for business use. 
# MAGIC 
# MAGIC $$ P(Delay) = \frac{\Sigma{Delayed \ Flights}}{\Sigma{All \ Flights}} $$

# COMMAND ----------

def flip(p_delay):
    """Determine positive or negative with given probability"""
    return 1 if random.random() < p_delay else 0

flip_udf = udf(flip, IntegerType())

# COMMAND ----------

# Baseline Model
class BaselineModel: 
    
    def __init__(self):
        return
        
    def fit_model(self, df): 
        """Calculate probability of delay with given dataframe"""
        count_delay = df[df["DEP_DEL15"] == 1.0].count()
        count_no_delay = df[df["DEP_DEL15"] == 0.0].count()

        p_delay = count_delay/(count_delay + count_no_delay)
        
        return p_delay
    
    def test_model(self, df, p_delay): 
        """Add prediction column by generating delay with calculated probability"""
        df_out = df.withColumn("predictions", flip_udf(lit(p_delay))).withColumn("rawPrediction", col("predictions")*1.0).withColumn("probability", struct( lit(p_delay), lit(1 - p_delay)))
        return df_out
    
    def evaluate_predictions(self, df, evaluator):
        """Plot AUC and ROC for baseline model"""
        
        auc = evaluator.evaluate(df)
        
        preds_for_plot = df.select("DEP_DEL15","probability").rdd.map(lambda row: (float(row['probability'][1]), float(row["DEP_DEL15"])))
        points = CurveMetrics(preds_for_plot).get_curve('roc')

        plt.figure()
        x_val = [x[0] for x in points]
        y_val = [x[1] for x in points]
        plt.title("ROC")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.plot(x_val, y_val)
        
        print("AUC: " + str(auc))
        
    def run_baseline_model(self, train_df, test_df): 
        """Main method to run the baseline model"""
        prob_delay = self.fit_model(train_df)
        predictions = self.test_model(test_df, prob_delay)
        evaluator = BinaryClassificationEvaluator(labelCol="DEP_DEL15")
        self.evaluate_predictions(predictions, evaluator)

# COMMAND ----------

baseline_model = BaselineModel()
baseline_model.run_baseline_model(df_airlines_weather, df_airlines_weather_19)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Windowing
# MAGIC 
# MAGIC As discussed in the [join method](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1858507102386329/command/1858507102387901) above, the weather readings for a 2 hour window, from 4 hours prior to departure to 2 hours prior to departure, were considered for each flight record. Therefore, the joined dataset had duplicates for each flight. Every weather reading inside the window would result in a row in the dataframe where the airline information was identical, but the weather readings would differ. Therefore, an aggregation had to be performed for each feature in order to produce a single record for each flight. This also meant that each feature needed to have its own aggregation method, that made sense for that feature. 
# MAGIC 
# MAGIC <br>
# MAGIC 
# MAGIC For the features from the airline dataset, including origin and destination, number of flights at origin, number of arrivals delayed, etc., these were duplicated across the rows. Therefore, a max() function was used to preserve the same value. 
# MAGIC 
# MAGIC <br>
# MAGIC 
# MAGIC For the numeric features from the weather dataset, an avg() function was used, to select the average weather reading in the 2 hour window. It was thought that this would provide the best feature to train the models on, as extreme minimums and maximums within the 2 hour window may have biased the feature one way or another. 
# MAGIC 
# MAGIC <br>
# MAGIC 
# MAGIC Finally, for the categorical features from the weather dataset, a first() function was used. These categorical features were saved in sparse one hot encoded vectors. The first() aggregation function would take the first sparse vector encountered for that flight. This is making the assumption that the weather conditions inside the one hot encoding vectors did not significantly vary inside the two hour window. This assumption was made because the EDA above showed that the airline features would be stronger indicators of delay and a general idea of weather conditions from one of the one hot encoded vectors would be sufficient for training and testing. 
# MAGIC 
# MAGIC <br>
# MAGIC 
# MAGIC With the methods described above, the dataset was reduced to one record per flight. The function used to perform this is included below. 

# COMMAND ----------

def simplify_dataframe_full(df, ohe=False, encoder=None):
    """
    Method to reduce dataframe to one record per flight after windowing was applied in the join. 
    
    Input: dataframe, boolean to one_hot_encode airline features, encoder to utilize
    Output: simplified dataframe, encoder utilized
    """
    
    #Drop record where label is missing. 
    #These typically indicate cancelled flights, which are not relevant to the business question. 
    df = df.where((df.DEP_DEL15).isNotNull())
    
    #Regression models require categorical features to be one hot encoded. 
    if ohe: 
        ohe_columns = ["DEP_HOUR","Day_Of_Week", "Month", "isHoliday"]
        ohe_out_columns = ["DEP_HOUR_OHE", "Day_Of_Week_OHE", "Month_OHE", "isHoliday_OHE"]
        #if encoder is not provided, this is a training. Create an encoder. 
        #If encoder is provided, this is a test. Use provided encoder
        if not encoder:
            encoder = OneHotEncoder(inputCols=ohe_columns, outputCols=ohe_out_columns).fit(df)

        #encode features
        df = encoder.transform(df)
        
        #preserve column names once OHE is done. 
        #Therefore, aggregation can be performed on same column names with or without encoding. 
        df = df.drop(*ohe_columns)
        for i in range(len(ohe_columns)):
             df = df.withColumn(ohe_columns[i], col(ohe_out_columns[i]))
        df.drop(*ohe_out_columns)
        
    #Do not select the columns that were only useful in joining. Lat-long, station id, etc...
    relevant_columns = ["FL_DATETIME_UTC", "DEP_HOUR_UTC", "OP_CARRIER_FL_NUM", "ORIGIN", "DEST", "DEP_HOUR", "Day_of_Week", "Month", "Distance", "number_dep_delay_origin", "number_flights_origin", "number_arr_delay_tail", "number_flights_tail", "percent_dep_delay_origin", "percent_arr_delay_tail", "isHoliday", "ELEVATION", "WIND_GUST_SPEED_RATE", "WIND_DIR_IMP", "WIND_SPEED_RATE_IMP", "VIS_DISTANCE_IMP", "CEILING_HEIGHT_IMP", "AIR_TEMP_IMP", "DEW_POINT_TEMPERATURE_IMP", "SEA_LEVEL_PRESSURE_IMP", "ATMOS_PRESSURE_3H_IMP", "PRECIPITATION_RATE", "PRESENT_ACC_OHE", "PAST_ACC_OHE", "ATMOS_PRESSURE_TENDENCY_3H_OHE", "WIND_TYPE_OHE", "DEP_DEL15"]

    #Reduce columns
    simplified_dataframe = df.select([col for col in relevant_columns])

    #Aggregate rows
    simplified_dataframe = simplified_dataframe.groupBy(simplified_dataframe.FL_DATETIME_UTC, simplified_dataframe.DEP_HOUR_UTC, simplified_dataframe.OP_CARRIER_FL_NUM, simplified_dataframe.ORIGIN) \
                                   .agg(max("DEST").alias("DEST"), \
                                        max("DEP_HOUR").alias("DEP_HOUR"), \
                                        max("Day_of_Week").alias("Day_of_Week"),\
                                        max("Month").alias("Month"),\
                                        max("Distance").alias("Distance"), \
                                        max("number_dep_delay_origin").alias("number_dep_delay_origin"), \
                                        max("number_flights_origin").alias("number_flights_origin"),\
                                        max("number_arr_delay_tail").alias("number_arr_delay_tail"),\
                                        max("number_flights_tail").alias("number_flights_tail"),\
                                        max("percent_dep_delay_origin").alias("percent_dep_delay_origin"),\
                                        max("percent_arr_delay_tail").alias("percent_arr_delay_tail"),\
                                        max("isHoliday").alias("isHoliday"),\
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
    return simplified_dataframe, encoder

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training, Testing, and Cross Validation
# MAGIC 
# MAGIC Since the problem involves temporal data, the dataset will need to be split in a way to prevent data leakage. This is to prevent previous information about a flight, such as flight ids, airplane codes, etc. in the train data being used to predict a flight delay rather than the features such as weather and airport conditions being used. Therefore, a split will have to be taken with the time of each record considered, to ensure that the test data is always after the train data. The strategy utilized for this was a Blocked TimeSeries Split [9]. The dates present in the dataset were divided into N blocks, were N indicates the number of desired folds. Records are assigned into each block where Block 1 has the earliest dates and Block N has the latest dates. Within each block, a train-test split is also generated. In the final pipeline, 80% of the data was used for training and 20% was used for testing. Therefore, the earliest 80% of days in the block are assigned to the training set and the latest 20% of days in the block are assigned to the test set. This produces N folds to be used for Cross Validation and training where there is no data leakage of temporal values. A visual representation is included below. 
# MAGIC 
# MAGIC In order to perform Cross Validation, a custom Cross Validation method was used [10]. This method takes in a dictionary of N dataframes, depending on the number of folds required.  A column in each dataframe indicates whether the record belongs to the train set or the test set. The custom Cross Validator also accepted a pyspark ParamGrid. This allowed for hyperparameters such as number of trees, depth of trees, type of regularization, etc. to be tested against. Each set of hyperparameters would be tested N times to determine the optimal set of hyperparameters. This custom Cross Validator then returned the optimal model.
# MAGIC 
# MAGIC The code used to assign folds, train and test sets and perform fitting with the custom cross validator is included in the cell below. 

# COMMAND ----------

# MAGIC %md
# MAGIC ![BlockTimeSeriesSplit](https://miro.medium.com/max/1210/1*QJaeOqGfe_vKbpmT882APA.png)
# MAGIC 
# MAGIC [9]

# COMMAND ----------

def fold_train_test_assignment(dataframe, folds, train_weight, features_col = "features", standardize=False, scaled_features_col="scaled_features"):
    """
    Method to assign dataframes to folds and train / test sets. 
    Input: dataframe, number of folds, training weight to use, column where to find features, boolean to indicate standardization, column to place features after standardizing
    Output: dictonary of dataframes (Number of entries is number of folds. Column in dataframe indicates train / test), scaler used. 
    """

    #Determine date range fo the dataframe
    max_date = dataframe.groupBy().agg(max("FL_DATETIME_UTC")).collect()[0]['max(FL_DATETIME_UTC)']
    min_date = dataframe.groupBy().agg(min("FL_DATETIME_UTC")).collect()[0]['min(FL_DATETIME_UTC)']
    
    prev_cutoff = min_date
    processed_df = {}
    
    #Create provided number of folds
    for i in range(folds):
        
        #Determine how many dates in each fold to have even sizing
        cutoff_date = min_date + ((max_date - min_date) / folds) * (i+1)
        
        #If this is the last fold, include the final record used as cutoff point. Otherwise, that record goes in the next fold. 
        if i+1 == folds: 
            cur_fold_df = dataframe.where((dataframe.FL_DATETIME_UTC >= prev_cutoff) & (dataframe.FL_DATETIME_UTC <= cutoff_date))
        else: 
            cur_fold_df = dataframe.where((dataframe.FL_DATETIME_UTC >= prev_cutoff) & (dataframe.FL_DATETIME_UTC < cutoff_date))
        
        #Determine cutoff date for train / test
        train_cutoff = prev_cutoff + (cutoff_date - prev_cutoff) * train_weight
        
        #label each row train or test based on date
        cur_fold_df = cur_fold_df.withColumn("cv", when(col("FL_DATETIME_UTC") <= train_cutoff, "train").otherwise("test"))
        
        std_scaler = None
        scaled_train = None
        #Regression models require features to be scaled
        if standardize: 
            #split into train and test
            train_df = cur_fold_df.where(cur_fold_df.cv == "train")
            test_df = cur_fold_df.where(cur_fold_df.cv == "test")
            #Create the scaler
            std_scaler = StandardScaler(inputCol=features_col, outputCol=scaled_features_col, withStd=True, withMean=True)

            #Fit the scaler on train data
            scaled_train = std_scaler.fit(train_df)
            scaled_train_df = scaled_train.transform(train_df)

            #Scale the test data
            scaled_test_df = scaled_train.transform(test_df)

            #combine into one dataframe again
            cur_fold_df = scaled_train_df.union(scaled_test_df)
        
        #Assign to dictionary
        processed_df["df" + str(i+1)] = cur_fold_df
        
        #Start point for next fold is end point of previous fold
        prev_cutoff = cutoff_date
    
    return processed_df, scaled_train

# COMMAND ----------

def fit_optimal_model(classifier, dataframe, feature_count, ranked_features, evaluator, grid, train_weight = 0.8, features_col = "features", folds = 5, standardize=False, scaled_features_col="scaled_features", reduce_features=True):
    """
    Method for cross validating using CustomCrossValidator and finding the optimal model 
    Input: classifier to use, dataframe to run on, number of features to use, features ranked by importance, evaluator to use, parameter grid to search, percent of rows to use for train, location of features column, number of folds to use, boolean if standardization required, where to place standardized features
    Output: optimal model, scaler used for standardization
    """

    #cut features down by importance to given number 
    if reduce_features: 
        #Select top features
        features_to_use = ranked_features[0:feature_count]
        feature_idx = [x for x in features_to_use['idx']]
        
        #Create column of reduced features
        slicer = VectorSlicer(inputCol=features_col, outputCol="features_reduced", indices=feature_idx)
        reduced_df = slicer.transform(dataframe)
        
        #Set classification column to reduced column
        classifier.setFeaturesCol("features_reduced")
        print("Reduced to optimal features...")
    else: 
        #Rename column so that method can be used with or without feature reduction
        reduced_df = dataframe.withColumn("features_reduced", col(features_col))
    
    #Assign folds and train / test sets
    dfs, scaler = fold_train_test_assignment(reduced_df, folds, train_weight, features_col="features_reduced", standardize=standardize, scaled_features_col=scaled_features_col)
    print("Created folds...")
    
    #Set column if standardization used
    if standardize:
        classifier.setFeaturesCol(scaled_features_col)
    
    #Run CustomCrossValidator and return optimal model
    cv = CustomCrossValidator(estimator=classifier, estimatorParamMaps=grid, evaluator=evaluator, splitWord = ('train', 'test'), cvCol = 'cv', parallelism=4)
    return cv.fit(dfs), scaler


# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Selection
# MAGIC 
# MAGIC After feature engineering and one hot encoding, there were 147 features to be utilized in the models. Initial efforts to use all of these features showed issues with generalization, where the scores for training models were much higher than scores for testing models. In order to reduce the amount of features, particularly for tree based models, feature importance was used. The feature importances generated by the pyspark classifieres indicated the predicitive power of each feature. It showed that the features from the airlines dataset were much more important to the model than features around weather. This aligns with the EDA, in that most flights were delayed due to airport or air traffic issues, which are captured in the airlines dataset. 
# MAGIC 
# MAGIC After investigating the feature importances, the ideal number of features to be used needed to be determined. This was trained as a hyperparameter. Different models were trained using different numbers of features and the one that performed best on a test dataset was used. For the final pipeline, which utilized a Random Forest model, only 40 features out of 147 had a non-zero feature importance score. The bottom 10 features also had very small feature importances. Therefore, the hyperparameter tuning was performed with 30, 10, and 5 features on the Random Forest model. 10 features proved to be the best performing model. The features it utilized in order of importance were: 
# MAGIC 
# MAGIC <br>
# MAGIC 
# MAGIC 1. Number of delayed arrivals for the given tail number between 4 hours to 2 hours prior to scheduled departure. 
# MAGIC 1. Percent of delayed arrivals vs on time arrivals for the given tail number between 4 hours to 2 hours prior to scheduled departure. 
# MAGIC 1. Percent of delayed departures vs on time departures for the given origin airport between 4 hours to 2 hours prior to scheduled departure. 
# MAGIC 1. Scheduled departure hour of the flight in local time
# MAGIC 1. Number of delayed departures for the given origin airport between 4 hours to 2 hours prior to scheduled departure. 
# MAGIC 1. Average ceiling height between 4 hours to 2 hours prior to scheduled departure
# MAGIC 1. Average dew point temperature between 4 hours to 2 hours prior to scheduled departure
# MAGIC 1. Average visibility distance between 4 hours to 2 hours prior to scheduled departure
# MAGIC 1. Average air temperature between 4 hours to 2 hours prior to scheduled departure
# MAGIC 1. Average sea level pressure between 4 hours to 2 hours prior to scheduled departure
# MAGIC 
# MAGIC The functions used to perform this testing are in the cell below. The results of the final pipeline are discussed in the sections below. 

# COMMAND ----------

def ExtractFeatureImp(featureImp, dataset, featuresCol):
    """
    Helper method to convert indicies to names for feature importances. 
    Input: Feature importance indicies, dataset used to generates, column where to find features. 
    Output: Dataframe with each row as (feature index, feature name, feature importance score)
    """
    list_extract = []
    #Iterate through columns in dataset and extract names by index
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]

    #convert to dataframe
    varlist = pd.DataFrame(list_extract)

    #Add score to each row
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])

    #return dataframe sorted by score
    return(varlist.sort_values('score', ascending = False))


def determine_optimal_feature_count(classifier, dataframe, feature_counts, evaluator, train_weight = 0.8, features_col = "features"):
    """
    Method to determine the optimal number of features to use for the given classifier.
    Input: classifier to use, dataframe to evaluate on, list of how many feautures to use, evaluator to compare models, weight to split df into train/test, column to find features
    Output: integer indicating ideal feature_count, dataframe of ranked features. 
    """
    
    #Create train / test split. Only 1 fold utilized for feature importance. 
    split_df = fold_train_test_assignment(dataframe, 1, train_weight)[0]['df1']
    train_df = split_df.where(split_df.cv == 'train')
    test_df = split_df.where(split_df.cv == 'test')
    print("Determined train / test split...")
    
    #Train model and rank features. 
    model = classifier.fit(train_df)
    ranked_features = ExtractFeatureImp(model.featureImportances, split_df, features_col)
    print("Ranked features by importance...")
    
    best_score = -1
    best_count = -1
    
    #Iterate through the given counts of features to test
    for count in feature_counts:
        #Select the top features
        features_to_use = ranked_features[0:count]
        feature_idx = [x for x in features_to_use['idx']]
        
        #create new column of top features
        slicer = VectorSlicer(inputCol=features_col , outputCol="features_reduced", indices=feature_idx)
        reduced_df_train = slicer.transform(train_df)
        reduced_df_test = slicer.transform(test_df)
        classifier.setFeaturesCol("features_reduced")

        #fit/test model using only top features
        cur_model = classifier.fit(reduced_df_train)
        cur_prediction = cur_model.transform(reduced_df_test)
        
        #Evaluate model
        cur_score = evaluator.evaluate(cur_prediction)
        print("AUC using " + str(count) + " features : " + str(cur_score))

        #Track the best AUC found and which count achieved it
        if cur_score > best_score:
            best_score = cur_score
            best_count = count

    print("Best AUC result achieved using " + str(best_count) + " features : " + str(best_score))
    return best_count, ranked_features
    

# COMMAND ----------

# MAGIC %md
# MAGIC ## Models Explored
# MAGIC 
# MAGIC Before settling on the Random Forest model that was used as the final pipeline below, a variety of other models were tried. This section will discuss the models that were tried, the justification, the preliminary results, and the descision to not include the model. 
# MAGIC 
# MAGIC ### Decision Tree
# MAGIC 
# MAGIC This model was chosen for the initial benefits of decision tree models. The database had features with heavy skew as well as a slew of categorical features. A decision tree model would be able to handle these hurdles, while other classifiers would have required additional preprocessing in order to train and test. However, the decision tree proved to have issues with generalization when using all the available features. Ultimately, it was abandonded in favor of the Random Forest model that generalized better. The results for this model are not included as its exploration was ended before all features were engineered, and thus does not provide an adequate comparison. 
# MAGIC 
# MAGIC ### Logistic Regression
# MAGIC 
# MAGIC For a two class classification problem, Logistic Regression appears as a natural choice for a classifier. However, this model would require some addtional preprocessing that was not included above. Firstly, regression models are unable to handle categorical variables. Therefore, categorical variables from the airlines dataset, such as depature hour, day of week, and month amoung others, had to be one hot encoded. Secondly, due to the large skew in the dataset, a scaler had to be used to standardize the features. Otherwise, the performance of the model would have been compromised, as Logistic Regression is sensitive to outliers and features on varying scales. Finally, the Logisitc Regression Classifer from pyspark does not provide a feature importance attribute. Therefore, the above method of limiting which features to use could not be utilized for this model, and all 147 features had to be used during cross validation. Ultimately, the lack of feature importance led to the decision to use Random Forest over this model, as it was more efficient to limit the number of features utilized for testing and prediction. 
# MAGIC 
# MAGIC The best AUC for this model was 0.814. This was achieved using a 5 fold cross validation with a grid search of 2 parameters (Regularization mode and aggregation depth) with 3 candidates each. The optimal parameters were 0.5 for regularization mode, indicating an even split between L1 and L2 regularization, and a aggregation depth of 8. Training for this model took 1.86 hours. This included \\( 2 parameters * 3 candidates * 5 folds = 30 models \\) trained and evaluated, of which 1 was selected. Running the 2019 data through the model to output the final scores took 1.80 minutes. 
# MAGIC 
# MAGIC ### Weighted Logistic Regression
# MAGIC 
# MAGIC This was an enhancement on the Logistic Regression that was done earlier with under-sampled majority class. Instead of under-sampling the majority class we added a "WEIGHT" column where the weight was computed that is inversely proportional to the occurance of each class. Specifically, the formula used is: 
# MAGIC 
# MAGIC $$ weight_{delay} = \frac{counts_{delay} + counts_{noDelay}} {2 * counts_{delay}}$$
# MAGIC $$ weight_{noDelay} = \frac{counts_{delay} + counts_{noDelay}} {2 * counts_{noDelay}}$$
# MAGIC 
# MAGIC The "WEIGHT" column is then given as the feature weights to be used for each row when the Logistic Regression runs. The same weights calculated at the training stage (2015 to 2018 data) are used in the 2019 data to get the results.
# MAGIC 
# MAGIC The best AUC for this model was 0.5. This was achieved using a 5 fold cross validation with the same grid search as the Logistic Regression Model. The optimal parameters were 0 for regularization mode, indicating L2 regularization, and an aggregation depth of 4. Training for this model took 28.81 minutes. Running the 2019 data through the model to output final scores took 2.11 minutes. This model did not provide significant benefit over the baseline, so it was abandonded. 
# MAGIC 
# MAGIC ### Gradient Boosted Trees (GBT)
# MAGIC A Decision Tree creates a model similar to a flow-chart where the leaf node is the decision (classification) when starting with a feature and making a branching decision at each stage. Starting from the root node the model learns to partition based on the feature value. With boosting the next stage learns from the mistake of the previous stage. Boosting uses weak learners, which are only slightly better than a random choice, to keep track of correct choices at every step. GBT uses a loss function instead of weighted average of individual outputs. The loss function is optimized using gradient descent. 
# MAGIC 
# MAGIC 
# MAGIC The model used a tree depth of 5. Anything beyond 5 took longer time and didn't improve the F1 score. The best F1-score from the was 0.74. The model took a relatively short time to train - tad lower than 5 minutes. This model was not run in cross-validation mode. Random Forest and Linear Regression gave better F1-scores and more efforts were focused to fine-tune these models. When GBT was run against 2019 data a F1-score of 0.7 was prodcued. Given these results the focus was shifted to Random Forest and Linear Regression. 
# MAGIC 
# MAGIC ### XG Boost (XGB):
# MAGIC In its simplest form XGB is a variant of GBT describe above that is optimized for speed. In GBT the stages of decision trees are built in parallel, while in XGB the trees are built in parallel. 
# MAGIC 
# MAGIC With XGB attempts with both the under-sample majority class and using weights as decribed in the **Weighted Logistic Regression** above were used. In both cases XGB gave a best case F1 score of 0.74, identical to that of a GBT. While XGB did run faster the F1-score was identical to that of the GBT. Again, as in the GBT case, cross-validation was not performed given that Random Forest and Logistic Regression showed better performance. 
# MAGIC 
# MAGIC The XGB took a little over 4 minutes to train on the 2015-2019 data, and complete the run with 2019 data in 30s. 
# MAGIC 
# MAGIC Please refer to the follow notebooks for additional details:
# MAGIC [full_modelling_pipeline_XGB_GBT_wtLR](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/2778166635284849/command/2778166635284886)
# MAGIC 
# MAGIC ### Note:
# MAGIC > For the XGB and the GBT models the AUC scores weren't calculated as the models didn't generate enough interest compared to other models (Random Forest and Logistic Regression). 
# MAGIC 
# MAGIC ### Ensemble
# MAGIC Stacking Ensemble involves combining the predictions from multiple machine learning models (base models) on the same dataset and building another model (meta model) that predicts which of these models to use while making predictions. Two different [stacking ensembles](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/4070574710010114/command/4070574710011577) made of two levels, with the predictions of the level 0 models being the features for the level 1 model were implemented. Both stacking ensembles had Random Forests, Logistic Regression and Gradient boost classifiers for level 0 models, but one of the stacking ensemble had a XGBoost classifier for the level 1 where as the other stacking ensemble had a Logistic Regression classifier.
# MAGIC 
# MAGIC The level 0 models were setup with a pipeline consisting of stages to compute the average weather readings for the flight from the measurements recorded during the weather window, to down-sample the training and test sets since the delayed and on-time flights were imbalanced, and to convert the features to a vector. A custom cross validator with 5 folds to train the models was used. The stacking ensemble having XGBoost as level 1 model obtained a F1-Score of 0.5860 on the 2019 test data, which was much lower than the F1-scores of the level 0 models. The stacking ensemble having Logistic Regression as level 1 model obtained a F1-score of 0.6459 on the 2019 test data, improving over the F1-scores of the level 0 models.
# MAGIC 
# MAGIC The level 0 logistic Regression model took 17 minutes to train, while the Gradient Boost took 42 minutes to train. The level 1 XGBoost model took 12.65 minutes and the level 1 Logistic Regression took 4.49 minutes to train.

# COMMAND ----------

def plots_for_models(prediction_dfs): 
    """
    Generate plots for predictions from various models
    Input: Dict of dataframes to generate plots from. 
    """
 
    #Create grid for plots
    fig, axs = plt.subplots(ncols=len(prediction_dfs), figsize=(40,10))
    
    for i in range(len(prediction_dfs)): 
        name = list(prediction_dfs.keys())[i]
        df = prediction_dfs[name]
        
        #Select probability and labels to generate ROC curve
        probability_col = 'probability'
        preds_for_plot = df.select("DEP_DEL15",probability_col).rdd.map(lambda row: (float(row[probability_col][1]), float(row["DEP_DEL15"])))
        points = CurveMetrics(preds_for_plot).get_curve('roc')

        x_val = [x[0] for x in points]
        y_val = [x[1] for x in points]

        #Plot ROC curve
        axs[i].set_title("ROC for " + name)
        axs[i].set(xlabel="False Positive Rate", ylabel="True Positive Rate")
        axs[i].plot(x_val, y_val, )

    

predictions_lr = spark.read.parquet(f"{blob_url}/predictions/lr_optimized")
predictions_lr.cache()
predictions_lrw = spark.read.parquet(f"{blob_url}/predictions/lrw_optimized")
predictions_lrw.cache()
predictions_xgb = spark.read.parquet(f"{blob_url}/predictions/xgb_optimized")
predictions_xgb.cache()
predictions_gbt = spark.read.parquet(f"{blob_url}/predictions/gbt_optimized")
predictions_gbt.cache()
predictions_ensemble = spark.read.parquet(f"{blob_url}/predictions/ensemble")
predictions_ensemble.cache()
dict_to_plot = {}
dict_to_plot['Logistic Regression'] = predictions_lr
dict_to_plot['Weighted Logistic Regression'] = predictions_lrw
dict_to_plot['XG Boost'] = predictions_xgb
dict_to_plot['Gradient Boosted Tree'] = predictions_gbt
dict_to_plot['Ensemble'] = predictions_ensemble
plots_for_models(dict_to_plot)

# COMMAND ----------

# MAGIC %md
# MAGIC # Final Pipeline
# MAGIC 
# MAGIC The final model chosen was a Random Forest model. This model provided a AUC measure of 0.811. There were several steps taken to prepare and tune this model for this result, and the details are discussed below. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preperation 
# MAGIC 
# MAGIC The first step in data preparation was the aggregation of records inside the 2 hour window from 4 hours to 2 hours prior to departure. The method and code used for this has been discussed [above](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1858507102386329/command/1858507102387959). Since this is a tree based model, the categorical features did not have to be one hot encoded, so that branch in the function was not executed. 
# MAGIC 
# MAGIC The next step required formatting the data in order to prepare for useage in the pyspark modeling library. This involves using a VectorAssembler to create a sparse vector representation of all the features in each row. This column can then be passed to the modeling library. Due to sparse representation, less data needs to be passed between nodes in the Spark cluster while training and testing. 
# MAGIC 
# MAGIC The final step to prepare the data was downsampling. Since there is a large discrepancy between non delayed and delayed flights, with non delayed being the majority class, downsampling was used during training to have a balanced model. Without downsampling, there were issues with generalization, as the model heavily biased toward prediciting no delay. 
# MAGIC 
# MAGIC The assembling and downsampling code is included below. After these steps are completed, the data is prepared for modeling. 

# COMMAND ----------

def prepare_dataframe_full(df, include_wt=False, wt=(0,0), label_col="DEP_DEL15"):
    """
    Method to include features into a single sparse vector. This format is required for pyspark classifiers. 
    Input: Dataframe, boolean to include weight column, weights to use, column where to find labels
    Output: Dataframe with feature column included, weights used
    """
    
    #Select all columns needed for features
    feature_columns = ["DEP_HOUR", "Day_of_Week", "Month", "Distance", "number_dep_delay_origin", "number_flights_origin", "number_arr_delay_tail", "number_flights_tail", "percent_dep_delay_origin", "percent_arr_delay_tail", "isHoliday","ELEVATION", "WIND_GUST_SPEED_RATE", "WIND_DIR_IMP", "WIND_SPEED_RATE_IMP", "VIS_DISTANCE_IMP", "CEILING_HEIGHT_IMP", "AIR_TEMP_IMP", "DEW_POINT_TEMPERATURE_IMP", "SEA_LEVEL_PRESSURE_IMP", "ATMOS_PRESSURE_3H_IMP", "PRECIPITATION_RATE", "PRESENT_ACC_OHE", "PAST_ACC_OHE", "ATMOS_PRESSURE_TENDENCY_3H_OHE", "WIND_TYPE_OHE"]

    #If using weight with this classifier, generate the weight column
    if include_wt:
        if wt == (0,0):
            count_delay = df[df[label_col] == 1.0].count()
            count_no_delay = df[df[label_col] == 0.0].count()
            #Compute weight by inverse proportion to the occurance of each class
            wt = ((count_delay + count_no_delay) / (count_delay * 2), (count_delay + count_no_delay) / (count_no_delay * 2))
        
        #Add weights to dataframe
        df = df.withColumn("WEIGHT", when(col(label_col) == 1.0, wt[0]).otherwise(wt[1]))
        print("Weight for delayed flights:", wt[0], "|Weight for non-delayed flights:", wt[1])
    
    #Add features to feature column
    assembler = VectorAssembler(inputCols = feature_columns, outputCol = "features").setHandleInvalid("skip")
    prepared_dataframe = assembler.transform(df)
    return prepared_dataframe, wt


def sample_dataframe(dataframe, delayed_percent=0.90, non_delayed_percent=0.20):
    """
    Helper method to downsample the majority class: non delayed flights
    Input: dataframe to downsample
    Output: downsampled dataframe
    """
    #split into classes
    delayed_df = dataframe.where(dataframe.DEP_DEL15 == 1)
    non_delayed_df = dataframe.where(dataframe.DEP_DEL15 == 0)
    
    #Select rows from each class. The percent values were pre computed to end up with a 50-50 split between classes
    delayed_df_samp = delayed_df.sample(False, delayed_percent, seed = 2018)
    non_delayed_df_samp = non_delayed_df.sample(False, non_delayed_percent, seed = 2018)

    #Combine and return
    return delayed_df_samp.union(non_delayed_df_samp)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Training and Testing
# MAGIC 
# MAGIC The Random Forest model was run through a multistep training process. A few steps of this process were disucssed above, but it is summarized here for clarity. 
# MAGIC 
# MAGIC 1. Determine optimal number of features. This is done by training a model, ranking the feature importances, and then training and evaluating 3 models using 30, 10, and 5 features. A single fold is used for this validation. The best performing model for Random Forest utilized 10 features as discussed above. This took 1 model to determine the importances and 3 models to evaluate for a total of 4 training exercises. 
# MAGIC 1. Cross Validation using the CustomCrossValidator and 5 folds. This is to tune hyperparameters and each parameter combination in the grid is run on all 5 folds. This is equaivalent to \\( 3 parameters * 3 options / parameter * 5 folds = 45 models \\). The total of 49 models, including the feature importances, took 8.86 hours to train and evaluate. For the final pipeline, the hyperparameters tested are (* = optimal number): 
# MAGIC  
# MAGIC     1. Maximum Depth of Trees: 20*, 25, 30
# MAGIC     1. Number of Trees: 20, 22, 25*
# MAGIC     1. Minimum Instances per Node : 35, 40*, 45
# MAGIC  
# MAGIC  
# MAGIC 1. Testing the model involved running the 2019 dataset through the aggregation and assembling steps in data preparation. The downsampling is skipped for testing. Testing this model on the 2019 dataset took 1.65 minutes. 

# COMMAND ----------

# MAGIC %md
# MAGIC The full pipeline of training, tuning, and testing the Random Forest model is available <a href="$./full_modelling_pipeline">here</a>. To aid understanding an example is included below. 
# MAGIC 
# MAGIC <br>
# MAGIC 
# MAGIC This example is run using the first 10 days of data from 2015. This data is from after the join is performed. To trace through the algorithm, a single flight is selected: Flight 55 departing from ATL to IAH on January 4th. Directly after the join, the flight has multiple records in the dataset. This is due to weather readings that were in the window of 4 hours prior to 2 hours prior to departure. The airline features, such as origin / dest, number_flights_origin, etc are identical in each row. The weather features, such as CEILING_HEIGHT and VIS_DISTANCE_IMP have differences. 

# COMMAND ----------

#Select first 10 days of data
example_df = df_airlines_weather.where(col("FL_DATE_UTC") < "2015-01-11")
example_df.cache()

#A particular flight to track
ex_datetime = "2015-01-04T14:20:00.000+0000"
ex_hour = 14
ex_fl_num = 55
ex_origin = "ATL"

#Display flight right after join
ex_flight = example_df.where((example_df.FL_DATETIME_UTC == ex_datetime) & (example_df.DEP_HOUR_UTC == ex_hour) & (example_df.OP_CARRIER_FL_NUM == ex_fl_num) & (example_df.ORIGIN == ex_origin))
display(ex_flight)

# COMMAND ----------

# MAGIC %md
# MAGIC The first step in the pipeline is to aggregate the records in the window to a single record. This is done using the method described in the [Windowing](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1858507102386329/command/1858507102387959) section above. The example flight has now been reduced to one record, with the airlines features intact, and the numeric weather features aggregated. The extraneuous columns used in the join, such as station and lat/long information have also been dropped. 

# COMMAND ----------

#Aggregate rows and drop extra columns
simp_ex_df = simplify_dataframe_full(example_df)[0]
simp_ex_df.cache()

#Display example flight after aggregation
simp_ex_flight = simp_ex_df.where((example_df.FL_DATETIME_UTC == ex_datetime) & (example_df.DEP_HOUR_UTC == ex_hour) & (example_df.OP_CARRIER_FL_NUM == ex_fl_num) & (example_df.ORIGIN == ex_origin))
display(simp_ex_flight)

# COMMAND ----------

# MAGIC %md
# MAGIC Next, the features are assembled into a sparse vector for use in modeling. Notice the example row now has a "features" column added to the end that contains all the relevant features. 

# COMMAND ----------

#Assemble features
prep_ex_df = prepare_dataframe_full(simp_ex_df)[0]
prep_ex_df.cache()

#Display example flight
prep_ex_flight = prep_ex_df.where((example_df.FL_DATETIME_UTC == ex_datetime) & (example_df.DEP_HOUR_UTC == ex_hour) & (example_df.OP_CARRIER_FL_NUM == ex_fl_num) & (example_df.ORIGIN == ex_origin))
display(prep_ex_flight)

# COMMAND ----------

# MAGIC %md
# MAGIC The final step in data preparation is down sampling the majority class. The data is skewed toward non delayed flights. Downsampling results in a much more even distribution of delayed to non delayed flights. 

# COMMAND ----------

#Downsample example dataframe
samp_ex_df = sample_dataframe(prep_ex_df, delayed_percent=0.80, non_delayed_percent=0.42)
samp_ex_df.cache()

#Counts before downsampling
prep_del = prep_ex_df.where(col("DEP_DEL15") == 1).count()
prep_no_del = prep_ex_df.where(col("DEP_DEL15") == 0).count()

#Counts after downsampling
samp_del = samp_ex_df.where(col("DEP_DEL15") == 1).count()
samp_no_del = samp_ex_df.where(col("DEP_DEL15") == 0).count()

#Display
print("Counts before sampling:")
print("Delayed: " + str(prep_del) + " Not Delayed: " + str(prep_no_del))

print("\nCounts after sampling:")
print("Delayed: " + str(samp_del) + " Not Delayed: " + str(samp_no_del))

# COMMAND ----------

# MAGIC %md
# MAGIC Now the modeling begins! The first step is the feature selection as described in the [Feature Selection](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1858507102386329/command/1858507102387955) section above. For this example, a model using the top 5 features produced the best AUC on test data. The ranked features are shown below and reiterate the importance of airport and airline features over weather features. 

# COMMAND ----------

#Counts to train on
feature_counts = [30, 10, 5]

#Classifier to use
rf = RandomForestClassifier(labelCol="DEP_DEL15", featuresCol="features", impurity="gini", seed=2018)

#Evaluator to use. Compares with AUC metric
evaluator = BinaryClassificationEvaluator(labelCol="DEP_DEL15")

#Find optimal feature count and feature importances
feature_count, ranked_features = determine_optimal_feature_count(rf, samp_ex_df, feature_counts, evaluator)
display(ranked_features)

# COMMAND ----------

# MAGIC %md
# MAGIC Next, the optimal model is found using the training, testing, and cross validation method described in [Training, Testing, and Cross Validation](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1858507102386329/command/1858507102386333) above. For the purposes of this example, the scale of the grid search and number of folds are reduced. The end result of this is the optimal model that can then be used for generating predictions. This is the end of the full testing pipeline. For the full model on the full dataset, the notebook is available <a href="$./full_modelling_pipeline">here</a> and the results are discussed in the next sections. 

# COMMAND ----------

#Grid search to perform
grid = ParamGridBuilder()\
        .addGrid(rf.maxDepth, [20])\ #Depth of each tree in Random Forest
        .addGrid(rf.numTrees, [25])\ #Number of trees in Random Forest
        .addGrid(rf.minInstancesPerNode, [35, 40])\ #Minimum number of instances to be considered a node
        .build()

#Get optimal model. Scaler is not used for this classifier. 
model_rf, scaler = fit_optimal_model(rf, samp_ex_df, feature_count, ranked_features, evaluator, grid, folds=2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results
# MAGIC 
# MAGIC The AUC for the optimal Random Forest model was 0.811. The ROC curve and confusion matrix are included below. 

# COMMAND ----------

# MAGIC %md
# MAGIC **Summary**
# MAGIC |||
# MAGIC |---|---|
# MAGIC |Classifier|Random Forest|
# MAGIC | No. of Trees| 25 |
# MAGIC | Max Depth | 20 |
# MAGIC | Min. Instances / Node | 40 |
# MAGIC | Folds used | 5 |
# MAGIC | Size of training set | 8M records |
# MAGIC | Area Under Curve | 0.811 | 
# MAGIC | Accuracy | 84.4% |

# COMMAND ----------

def plots_for_rf(): 
    """
    Generate plots for predictions from Random Forest model
    """
    #Load predictions from blob storage
    predictions_rf = spark.read.parquet(f"{blob_url}/predictions/rf_optimized")
    predictions_rf.cache()

    #Select probability and labels to generate ROC curve
    preds_for_plot = predictions_rf.select("DEP_DEL15",'probability').rdd.map(lambda row: (float(row['probability'][1]), float(row["DEP_DEL15"])))
    points = CurveMetrics(preds_for_plot).get_curve('roc')

    #Create 1x2 grid for plots
    fig, axs = plt.subplots(nrows=2, figsize=(20,20))
    x_val = [x[0] for x in points]
    y_val = [x[1] for x in points]

    #Plot ROC curve
    axs[0].set_title("ROC")
    axs[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate")
    axs[0].plot(x_val, y_val, )

    #Select predictions and labels for confusion matrix
    preds_and_labels = predictions_rf.select(['prediction','DEP_DEL15']).withColumn('label', col('DEP_DEL15').cast(FloatType())).orderBy('prediction').select(['prediction','label'])

    #Generate matrix
    metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
    cm = metrics.confusionMatrix().toArray()

    #Add labels for matrix
    df_cm = pd.DataFrame(cm, index = ["True Not Delayed", "True Delayed"],
                  columns = ["Predicted Not Delayed", "Predicted Delayed"])

    #plot matrix
    sns.heatmap(df_cm, annot=True, ax=axs[1], fmt='.0f', vmin=100000, vmax=3000000)

plots_for_rf()

# COMMAND ----------

# MAGIC %md
# MAGIC # Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ## Drawbacks of Model
# MAGIC 
# MAGIC The Random Forest confusion matrix was used to investigate and identify the potential issues and data patterns that may be contributing to the erroneous classification (False Positives / False Negatives). The error analysis [notebook](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1858507102395289/command/1858507102395290) captures the exercise of this error analysis in detail for the curious minds.
# MAGIC 
# MAGIC Some of the findings are summarized below:
# MAGIC 
# MAGIC * **CASE 1**: The false positives that were sampled for the model were the result of the data points being available at the time of the prediction. For example, in one case the model predicted that the flight would get delayed based on the data that the arriving flight (for the same tail num) was delayed. Despite this the outbound flight of interest actually departed on time, and the model made an incorrect prediction in this case. While this arrival delay feature cannot be dropped given its high feature importance, there arises a need to compute additional feature in this case, which would be an indicator for the minimum allowed resting time for a flight after the arrival for the given tail number that would not consititute a departure delay.
# MAGIC 
# MAGIC * **CASE 2**: On the other hand, there were also observed instances where there were no arrival or departure delays in the 2 hour window of interest and yet there was an actual delay in departing flight. This conflict with the model prediction seems to have a some correlation with extremely bad weather conditions in an entirely different part of the US which has a potential to cause delays due to the cascading effects. The model failed to connect the factors beyond the 2 hour window and beyond the given geography. This solicits the need to evaluate some form of delay network model based on graph based algorithm that would account for delays in carriers / air traffic networks.
# MAGIC   
# MAGIC * **CASE 3**: The model also performed poorly in cases where the arriving flights were a part of short duration trips, specially those with a window smaller than our chosen 2 hour window. Short duration trips for an aircraft for connecting flights can extend their arrival into the time frame after the model has already made the prediction. In those cases, the model performs its best based on the availabe data, but yet results in a false negative.
# MAGIC 
# MAGIC The above discussion points and some of the detailed analysis with the code snippets has been captured in the notebook linked above. The important key take away is the impact of these incorrect classification on the main objective of the business case. False positives would imply a poor customer experience even potentially implying that some customers may miss their flight because they arrived at the airport late due to the incorrect notification that was sent out to them based on the model prediction. A False Negative would impact the perceived reliability / trust on the prediction model itself because the model failed to predict occurance of delays.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Future Work
# MAGIC 
# MAGIC The work conducted on this project of predicting flight delays has been limited to roughly 4 weeks of effort, there are several possibilities that could be uncovered to continue the work in this area. The number of available research papers, kaggle posts, peer discussions and data science blogs (all referenced below) are an inspiration to continue the efforts on this problem statement. There are a few items from the wishlist that can taken up at the next opportunity. They are listed below.   
# MAGIC 
# MAGIC <p>
# MAGIC   
# MAGIC * The error analysis exercise needs to be extended across additional cases for identifying patterns in incorrect classifications. 
# MAGIC * Consider including additional features based on the learnings from error analysis such as MIN_REQUIRED_RESTING_TIME.
# MAGIC * Consider taking into account weather features for destination, specially for short duration flights. 
# MAGIC * Consider including weather forecast features for the 2 hours between the prediction time and the time leading to the actual departure.
# MAGIC * Consider adding features that would capture the relative delay tendency for a given tail number / airline / carrier.
# MAGIC * Consider evaluation of graph model based on delay network to take into account the propagation effects of delays due to airline, flight, or extreme weather conditions at different geographical locations across the nation.
# MAGIC * Consider re-evaluating Tree Based Models without applying One Hot Encoding knowing the potential downsides of OHE for such models.
# MAGIC * Consider building models specific to distinct geographies across US.
# MAGIC 
# MAGIC <p>
# MAGIC After incorporating additional features given below and running the stacking ensemble with Logistic Regression, there were considerable improvements in the performance of the model. The model achieved an AUC score of 0.85 and F1-score of .6837 on the test data. With further hyperparameter tuning and feature selection, the model performance may further improve.
# MAGIC   
# MAGIC * number_flights_carrier - the number of flights for the carrier in the 2 hour window 2 hours prior to a flight's scheduled departure.
# MAGIC * number_dep_delay_carrier - the number of flights of the carrier delayed in the 2 hour window 2 hours prior to a flight's scheduled departure.
# MAGIC * percent_dep_delay_carrier - the percentage of flights of the carrier delayed in the 2 hour window 2 hours prior to a flight's scheduled departure.
# MAGIC * resting_time_tail_num - the time between the actual arrival time of the flight from the last trip to the scheduled departure time of the flight. 

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion

# COMMAND ----------

# MAGIC %md
# MAGIC ## Challenges Faced
# MAGIC 
# MAGIC #### Imbalanced Data and SMOTE
# MAGIC 
# MAGIC > SMOTE stands for Synthetic Minority Oversampling Technique. When there's imbalanced data, as in the case where on-time departures are a little over 4 times that of delayed departures, the model is straved of  minority class data (delayed departure data, in this case) to effectively learn feature values that result in minority class being the predictor. Over-sampling is one way to prevent this problem - over-sample the minority and under-sample mjority. An alternate approach is to add weights inversely proportional to the number of instances of each class. 
# MAGIC 
# MAGIC > SMOTE selects a minority class instance, **a** at random and finds its k nearest minority class neighbors (we used k=5). Then choose a random instance **b** from these k nearest neighbors, draw a line between a and b, and choose synthetic instances as a convex combination of **a** and **b**. The implementation is in the [notebook](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1898361324252165/command/1898361324252166).
# MAGIC 
# MAGIC > Given the size of the dataset SMOTE calls took a significantly long time to run. Even after that the F1-score and the predictions didn't have any significant improvement. In fact, the F1-scores were very similar to that of adding weights and random under-sampling. A few papers have suggested that the combination of SMOTE and under-sampling performs better. Given that weights and random under-sampling gave similar results and SMOTE took a long time to run it was decided to stick with random under-sample of majority class and using weights.
# MAGIC 
# MAGIC #### Caching
# MAGIC 
# MAGIC > The dataframes that are generated either after reading (parquet) files or through transformation needed to be cached. While it's not necessary to cache every dataframe, those dataframes that are repeatedly referred to and samples taken needed to be cached. Not caching these dataframes led to inconsistent results when actions such as sampling and count calculations for accuracy were done. 
# MAGIC 
# MAGIC 
# MAGIC #### Data leakage
# MAGIC 
# MAGIC > When dataframes were joined to form the final analysis F1-scores were much higher. This led to the suspicion that the model was getting data from the future. The goal is to predict the flight departure 2 hours prior. Clearly, that can't include any data beyond two hours prior to depature. When the merge happened, inadvertently samples from time window up to departure hour were taken. A join was done again to fix this issue. The "window" function doesn't allow for a specification of a time window. Two windows, one from 4 hours prior to departure until departure and the second from 2 hours prior to departure and up to deapture were taken. The data from 2nd window was removed from the first window, to get data two hours prior to departure.
# MAGIC 
# MAGIC 
# MAGIC #### Debugging for models
# MAGIC 
# MAGIC > An inadvertent error was made when the calculated "WEIGHT" column was included in the set of vectorized features. The model produced F1-score of 0.99998!! This happened due to the inclusion of WEIGHT column as a feature, which allowed the model to look at the "WEIGHT" value  and decide if it's delayed or not. Removing the "WEIGHT" column fixed this issue. 
# MAGIC 
# MAGIC > Initial models considered weather parameters as the sole factor in flight delays. The EDA showed that while weather indeed is a factor in aircraft delay, there are others such as "Carrier" operations parameter, "NAS" (National Airspace System, administered by FAA) that includes Air Traffic Control, and "Late Airctaft" arrival were having significant impacts. The approach in later phases specifically focused on factoring these features into the model. Consequently, most of the models saw F1-Score close to 82%.
# MAGIC 
# MAGIC > In several cases the arrival delay (by Tail Number) did impact the departure time; however, the model incorrectly predicted departure delay in some cases where there was none. This points to an important feature that needs to be added to the model. If the incoming aircraft is delayed but the delay is below some threshold (determined by the time gap between the actual arrival of the aircraft and the scheduled departure of the same aircraft under different flight number) then this arrival delay doesn't impact the departure.
# MAGIC 
# MAGIC > The model doesn't work well for flights of short duration. Suppose that the model predicted no delay at 8AM when the flight hadn't even taken off from the previous airport, based on conditions prior to 8AM, then the duration of the flight being short may lead to scenarios of delayed departure, delayed arrival etc. and that can cause the model to fail. To generalize this, if we need to predict delays "t" hours prios to depature at an airport and suppose that the aircraft that needs to come to the airport of interest hadn't taken off and the flight duration is less than "t" hours then the model can't predict well. 
# MAGIC 
# MAGIC 
# MAGIC #### Missing data handling
# MAGIC 
# MAGIC > Missing data in the weather dataset was handled by taking the average of prior 720 samples. The 720 samples corresponded to two week's worth of data. Thus, missing weather values were taken as an average of the prior two week's reading.
# MAGIC 
# MAGIC > Subsequent to the merge of weather data and airport data there were missing values for some time windows. Depending on the type of data, the missing values were filled with max, average, or first, as described in this [notebook](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/4070574709999496/command/4070574709999501). 
# MAGIC 
# MAGIC 
# MAGIC #### What would be done differently next time
# MAGIC 
# MAGIC > Retain unused columns in the data frame: The unused columns were removed as a way to simplify the data frame. In retrospect, having these columns would have helped debugging efforts to find out why the model didn't predict some of the delays correctly would be easier. It would have obviated the need to look into the joined dataframe.
# MAGIC 
# MAGIC > One-Hot-Encoding (OHE): OHE was considered a good mechanism to tranform categorical variable to numerical values. OHE allowed for running models such as Logistic Regression. However, OHE can have negative impact when it comes to decision-tree models such as Random Forest (RF). Models like RF may see OHE columns as sparse columns and may eliminate these categorical columns
# MAGIC 
# MAGIC > In the Random Forest model that was created for this project the accuracy, ROC, and F1-scores were good enough that a re-run without OHE was not considered due to the limited time.
# MAGIC 
# MAGIC > Add a feature to reflect the minimum "gate-time" (minimum time at the gate from when the aircraft arrived and when the aircraft can take off) so that delayed arrival of an aircraft can be correctly modeled to see if it factors in the delayed departure.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Course Concepts Utilized
# MAGIC 
# MAGIC The course on Machine Learning at Scale (W261) stitched together many of the learnings from the MIDS coursework all together, and enabled the ability to think about approaching large scale data problems. While most of the learnings from this course influenced the problem solving for the Flight Delay Prediction problems, some of the key concepts that were applied during this project are below.
# MAGIC 
# MAGIC *Scalability* - Working with a large dataset as airlines and weather provided the opportunity to apply the learnings of this course directly in this project. The strategy for first performing initial EDA on a smaller subset of 2015 Q1/ Q2 data helped quickly building familiarity with the datasets. Performing the join on the full data set was seemingly the most time consuming step in this process, but the foundations laid and the mindset created for making large scale complex problems as embarassingly parallel as possible helped tackle some of the challenges. The exposure to solving this problem with Spark running on Databricks Infrastructure enabled the creation of an end to end machine learning pipeline while evaluating multiple models along with their prerequisite data exploration and feature engineering.
# MAGIC 
# MAGIC *Feature Selection* - The steps performed for the feature selection are discussed in this report earlier. The use of PCA, Lasso Regression and Feature Importance was also deliberated in the early stages of the project when the model selection had not yet been concluded. The Decision Tree and Random Forest models do offer the feature importance which were ultimately used for dropping the tail set of features that had very low feature importance scores. By iterating and plotting the model performance against the reduced set of features in each run, we were able to inspect and close in on the number of features that provided us the most optimal score for training and test data sets. This was then applied to the 2019 dataset to ensure that the model generalized well.
# MAGIC 
# MAGIC *Feature Encoding* - The datasets for airlines and weather included numeric as well as categorical features. The appropriate encoding of these variables is a preliminary requirement for training models. One Hot Encoding was applied for the categorical variables as applicable and the sparse encoded vectors were used for training different models
# MAGIC 
# MAGIC *Data Normalization* - Logistic Regression based model required rescaling the data using the StandardScalar API  

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final Thoughts 
# MAGIC The project tied the course work to a real-world problem that has a tangible impact on the customer experience, airline operations for both the carrier and airport operations, and for all other entities. The project showed the importance of analyzing data, making calls on missing, null values, and explicitly coded missig values. In fact, some of the data clean-ups and the subsequent joins of the resulting data was fairly complicated and involved long time to complete. The size of the data set is quite large that a couple of times the join had to be rerun with changes. These tasks did take a long effort, though this was manageable within the project period.
# MAGIC 
# MAGIC The project also revealed the importance of concepts such as imbalance handling, data normalization, feature encoding to handle categoricals etc. that were needed to run the models. Once the EDA, filling in missing values, and joins were done, running the model itself wasn't a huge task. The complexities involved in running the model included setting up the dataframe with the right set of features, under-sampling majority class/weighting the minority class higher, and generating the cross-validation parameters. The next big step was in the interpretation of the test results. Unlike "Sci-kit learn" some of the functions to visualize the ROC (Received Operating Characteristics) and PR (Precision-Recall) validation weren't easily available in PYSPARK MLLIB. These functions needed to be realized. Once the functions were realized then came the task of interpreting the results. Looking at any one score alone doesn't give the full picture. Looking at Precision, Recall, and F1-score as well as the accuracy gave an almost full picture of the modeling.
# MAGIC 
# MAGIC Critical to the project was the debug efforts to understand the False Positives and False Negatives. A key insight is that for flights of short duration (defined as the duration of the flight of the incoming aircraft) the model performs less efficiently. This is because of the fact that any prediction needs to consider all parameters 2 hours prior to depature. Thus, if the incoming aircraft hadn't taken off 2 hours prior to the scheduled departure of same aircraft (under a different flight number) then any delays impacting this aircraft is unknown to the model. Similarly, a delayed arrival of an aircraft doesn't always signify depature delay, though it's an important factor. The delay depends on the time elapsed between when the aircraft arrived and the scheduled departure of the next flight. The gate-time may be long enough to absorb the delay due to incoming aircraft. 
# MAGIC 
# MAGIC The Random Forest Model gave the best result among the models that were attempted so far, followed very closely by Logistic Regression. The best model achieved an accuracy of a little over 84% and an AUC (Area Under Curve) of 0.811 (with 1 being the theoretical maximum any model can get to). Overall, the journey from the EDA phase to arriving at a reasonably working model was full of learnings. The list of actions have been outlined in feature engineering as well modeling enhancements that could be performed to further improve the model and the results.
# MAGIC 
# MAGIC The stated research question at the beginning of the report was: _**will a flight's departure be delayed by 15 or more minutes, based on weather and airport conditions 2 hours prior to scheduled departure?**_. The significance of this problem statement cannot be under-estimated. The more accuracy with which we flight delays can be predicted, the more the model will help all the constituents - passengers, airline/carrier operation, Air Traffic Control, Airport Administration, gate crew, and all other stakeholders. Even if the financial implications associated with flight delays to the tune of $40B could be reduced by a few percent points, that would result a huge savings, and will benefit both the service providers and the consumers alike. The work presented in this report certainly has room to be further refined to make the model performance better. It's important to bear from a traveling public perspective False Negative (the case of a delayed aircraft predicted as not being delayed) is better than predicting a non-delayed flight as delayed. Clearly, False Positives and False Negatives have to be lowered in order to gain acceptance of the model, that one day powers a mobile application. 

# COMMAND ----------

# MAGIC %md
# MAGIC #References
# MAGIC 1. https://www.jec.senate.gov/public/index.cfm/democrats/2008/5/your-flight-has-been-delayed-again_1539
# MAGIC 1. https://transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr			
# MAGIC 1. https://www.bts.dot.gov/explore-topics-and-geography/topics/airline-time-performance-and-causes-flight-delays			
# MAGIC 1. https://www.bts.gov/topics/airlines-and-airports/understanding-reporting-causes-flight-delays-and-cancellations			
# MAGIC 1. https://www.transtats.bts.gov/printglossary.asp		
# MAGIC 1. [Imbalanced Learning: Foundations, Algoirthms, and Applications, 2013](https://amzn.to/32K9K6d)
# MAGIC 1. https://arxiv.org/abs/1106.1813
# MAGIC 1. https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc#:~:text=An%20ROC%20curve%20receiver%20operating,False%20Positive%20Rate
# MAGIC 1. https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4
# MAGIC 1. https://www.timlrx.com/blog/creating-a-custom-cross-validation-function-in-pyspark
# MAGIC 1. https://towardsdatascience.com/one-hot-encoding-is-making-your-tree-based-ensembles-worse-heres-why-d64b282b5769