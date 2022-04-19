# Databricks notebook source
from functools import reduce
from pyspark.sql import DataFrame
from pyspark.sql.functions import col,split,isnan, when, count, concat, lit, desc, floor, hour, sum, max, min, avg, first
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import SQLContext
from pyspark.sql import DataFrameNaFunctions
from pyspark.ml import Pipeline
from pyspark.ml.feature import Binarizer
from pyspark.ml.feature import OneHotEncoder, VectorAssembler, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator 
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.sql.types import FloatType
import matplotlib.pyplot as plt
import seaborn as sn

from pyspark.sql.types import FloatType

# COMMAND ----------

from pyspark.sql.functions import col, max

blob_container = "team08-container" # The name of your container created in https://portal.azure.com
storage_account = "team08" # The name of your Storage account created in https://portal.azure.com
secret_scope = "team08-scope" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "team08-key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# COMMAND ----------

df_airlines = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data/2019*")

# COMMAND ----------

display(df_airlines)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Objective
# MAGIC 
# MAGIC We evaluated a set of models for the predicting airline delays and have the best results with the Random Forest Model. The confusion matrix for the Random Forest is as shown below. The goal for this error analysis is to determine the potential causes of incorrect predictions (False Positive and False Negative).

# COMMAND ----------

## Reload the Random Forest Model
predictions_rf = spark.read.parquet(f"{blob_url}/predictions/rf_optimized")

# COMMAND ----------

# Plot the Confusion Matrix
preds_and_labels = predictions_rf \
                        .select(['prediction','DEP_DEL15']) \
                        .withColumn('label', col('DEP_DEL15').cast(FloatType())).orderBy('prediction') \
                        .select(['prediction','label'])

metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
cm = metrics.confusionMatrix().toArray()
df_cm = pd.DataFrame(cm, index = ["True Not Delayed", "True Delayed"], columns = ["Predicted Not Delayed", "Predicted Delayed"])

# COMMAND ----------

sn.heatmap(df_cm, annot=True, fmt='.0f', vmin=100000, vmax=3000000);

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### The Ocular Test

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Step 1 - Choose Features to Investigate
# MAGIC 
# MAGIC We computed the best score for RF with 10 features, so we will focus on the top 10 features for this investigation in an attempt to identify patterns. For Random Forest, the list of top 10 features have been computed [here](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/4070574709999496/command/4070574709999541). We will make use of the top 10 features based on their feature importance.

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC #### Step 2 - Examine False Positives Data Frame
# MAGIC 
# MAGIC Filter the top 10 features and the False Positives - Model identified that there would be a delay, but there was no delay.
# MAGIC 
# MAGIC There could be several reasons for False Positives. For predicting the delay for a flight scheduled for departure at 10 AM, we are making the prediction at 8 AM, based on the data that we have available in the 2 hours prior to 8 AM. 
# MAGIC 
# MAGIC <p>
# MAGIC Some reasons may span the following:
# MAGIC <p>
# MAGIC   
# MAGIC * A late arriving aircraft may still depart on time on the next leg.
# MAGIC * Poor weather conditions impacting delays may improve over the next 2 hours prior to scheduled departure.
# MAGIC * NAS delays may observe dwindling Air Traffic congestion over the next couple hours.
# MAGIC * A different flight (Tail Number) may be brought in by the airline to service the route.

# COMMAND ----------

pred_fp = predictions_rf.where((predictions_rf.prediction == "1") & (predictions_rf.DEP_DEL15 == "0")).select("FL_DATETIME_UTC", "OP_CARRIER_FL_NUM","DEP_HOUR_UTC","number_arr_delay_tail","percent_arr_delay_tail","percent_dep_delay_origin","DEP_HOUR","number_dep_delay_origin","CEILING_HEIGHT_IMP","DEW_POINT_TEMPERATURE_IMP","VIS_DISTANCE_IMP","AIR_TEMP_IMP","SEA_LEVEL_PRESSURE_IMP", "prediction", "DEP_DEL15")
display(pred_fp)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Step 3 - Examine False Negatives
# MAGIC 
# MAGIC Filter the top 10 features and the False Negatives - model identified that there would be no delay, but there was a delay. Again, for predicting the delay for a flight scheduled for departure at 10 AM, we are making the prediction at 8 AM, based on the data that we have available in the 2 hours prior to 8 AM. 
# MAGIC 
# MAGIC <p>
# MAGIC Some of the potential reasons why the model failed to predict a delay may include
# MAGIC <p>
# MAGIC   
# MAGIC * Weather conditions deteriorated within the 2 hour window, after we made the prediction.
# MAGIC * Sudden/ Unexpected events that occur within the 2 hour window prior to departure - the data for which was not available at the time of prediction.
# MAGIC * Flights that were not predicted to arrive late within the 2 hours after making a prediction, actually arrive late, thereby impacting the departure.   
# MAGIC   

# COMMAND ----------

pred_fn = predictions_rf.where((predictions_rf.prediction == "0") & (predictions_rf.DEP_DEL15 == "1")).select("FL_DATETIME_UTC", "OP_CARRIER_FL_NUM","DEP_HOUR_UTC","number_arr_delay_tail","percent_arr_delay_tail","percent_dep_delay_origin","DEP_HOUR","number_dep_delay_origin","CEILING_HEIGHT_IMP","DEW_POINT_TEMPERATURE_IMP","VIS_DISTANCE_IMP","AIR_TEMP_IMP","SEA_LEVEL_PRESSURE_IMP", "prediction", "DEP_DEL15")
display(pred_fn)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Step 4 - Cross check FP/FN against TP/TN
# MAGIC 
# MAGIC Filter the top 10 features and the TP and TN and evaluate if there are any patterns that differentiate these results from the FP / FN.

# COMMAND ----------

pred_tp_tn = predictions_rf.where(predictions_rf.prediction == predictions_rf.DEP_DEL15).select("FL_DATETIME_UTC", "OP_CARRIER_FL_NUM","DEP_HOUR_UTC","number_arr_delay_tail","percent_arr_delay_tail","percent_dep_delay_origin","DEP_HOUR","number_dep_delay_origin","CEILING_HEIGHT_IMP","DEW_POINT_TEMPERATURE_IMP","VIS_DISTANCE_IMP","AIR_TEMP_IMP","SEA_LEVEL_PRESSURE_IMP", "prediction", "DEP_DEL15")
display(pred_tp_tn)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Investigation #1 False Positive Case

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC #### Pick a record with false positive

# COMMAND ----------

display(pred_fp.where((pred_fp.FL_DATETIME_UTC == "2019-03-04T01:09:00.000+0000") & (pred_fp.OP_CARRIER_FL_NUM == 5483)))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Cross examine by fetching the flight details for the tail number associated with the above record 

# COMMAND ----------

display(df_airlines.where((df_airlines.FL_DATE <= "2019-03-04") 
                               & (df_airlines.FL_DATE >= "2019-03-03")
                               & (df_airlines.TAIL_NUM == "N554NN") \
                               ).select("CRS_DEP_TIME", 
                                        "TAIL_NUM", 
                                        "ARR_DEL15", "DISTANCE", "ORIGIN", "DEST", 
                                        "NAS_DELAY", "CARRIER_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY","SECURITY_DELAY", 
                                        "*") \
                                .sort("FL_DATE", "CRS_DEP_TIME"))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Analysis
# MAGIC 
# MAGIC <p>
# MAGIC   
# MAGIC * No issues detected in the join. We can confirm this because we see 2 flight entries as expected between 16:09 hours and 18:09 hours for this tail number for the flight departure at 20:09 hours. We also evaluated this across some other records.
# MAGIC * We also can see that this given flight was enroute from CLT to OAJ and then from OAJ to CLT on the two prior legs of its journey and was delayed both times within the two hours.
# MAGIC * The flight departure delay was predicted due to this reason. However, despite the two delays for the given tail number, the flight actually departed on time. 
# MAGIC * This signifies that there was a potential buffer between the arrival of the flight and its departure which covered for the arrival delay and did not impact the ON-TIME DEPARTURE.
# MAGIC * Add a feature that is a proxy for the minimum required resting time that does not impact on-time departure. 
# MAGIC * Feature *MIN_REQUIRED_RESTING_TIME* computed as ( scheduled_departure_time - actual_arrival_time of the previous leg )

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Investigation #2 False Positive Case

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Pick a record with FP

# COMMAND ----------

display(pred_fp.where((pred_fp.FL_DATETIME_UTC == "2019-03-04T17:11:00.000+0000") & (pred_fp.OP_CARRIER_FL_NUM == 7411)))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Cross check airlines table for given airport origin

# COMMAND ----------

df2 = df_airlines.where((df_airlines.FL_DATE <= "2019-03-04") 
                               & (df_airlines.FL_DATE >= "2019-03-03")
                               & (df_airlines.ORIGIN == "DTW") \
                               ).select("FL_DATE", "CRS_DEP_TIME", 
                                        "TAIL_NUM", "OP_CARRIER_FL_NUM",
                                        "ARR_DEL15", "DISTANCE", "ORIGIN", "DEST", "DEP_DEL15", 
                                        "NAS_DELAY", "CARRIER_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY","SECURITY_DELAY", 
                                        "*") \
                                .sort("FL_DATE", "CRS_DEP_TIME")
display(df2.where((df2.FL_DATE == "2019-03-04") & \
                   (df2.CRS_DEP_TIME >= 811) & \
                   (df2.CRS_DEP_TIME <= 1011 )
                 )
       )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Cross check airlines table for given tail number

# COMMAND ----------

df3 = df_airlines.where((df_airlines.FL_DATE <= "2019-03-04") 
                               & (df_airlines.FL_DATE >= "2019-03-03")
                               & (df_airlines.TAIL_NUM == "N349NW") \
                               ).select("FL_DATE", "CRS_DEP_TIME", 
                                        "TAIL_NUM", "OP_CARRIER_FL_NUM",
                                        "ARR_DEL15", "DISTANCE", "ORIGIN", "ORIGIN_STATE_NM", "DEST", "DEST_STATE_NM", 
                                        "NAS_DELAY", "CARRIER_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY","SECURITY_DELAY", 
                                        "*") \
                                .sort("FL_DATE", "CRS_DEP_TIME")
display(df3)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Analysis
# MAGIC 
# MAGIC * March 3rd and March 4th of 2019 had extreme weather conditions across large parts of the US, but not DTW / Michigan State.
# MAGIC * Finding: Potential impact of Severe Tornadoes on the flight delays
# MAGIC   * https://www.weather.gov/okx/20190303_04_WinterStorm
# MAGIC   * https://ny.curbed.com/2019/3/3/18249440/new-york-weather-winter-storm-2019-snow
# MAGIC   * https://thegate.boardingarea.com/updated-travel-alert-march-2019-winter-weather-affects-northeastern-and-rocky-mountain-united-states-and-southeastern-canada/
# MAGIC * Effect of poor weather condition in a part of the US and its impact on predicting the delay accurately
# MAGIC * How to avoid the false positive as assessed by the model in this case ?
# MAGIC * Potential solution is to apply and make use of delay network / graph model.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Investigation #3 False Negative Case

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Pick a record with FN

# COMMAND ----------

display(pred_fn.where((pred_fp.FL_DATETIME_UTC == "2019-07-25T18:38:00.000+0000") & (pred_fp.OP_CARRIER_FL_NUM == 5026)))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Cross Check flight number in airlines dataset

# COMMAND ----------

display(df_airlines.where((df_airlines.FL_DATE == "2019-07-25") &
                              (df_airlines.OP_CARRIER_FL_NUM == 5026) \
                               ).select("FL_DATE", "CRS_DEP_TIME", 
                                        "TAIL_NUM", "OP_CARRIER_FL_NUM",
                                        "ARR_DEL15", "DISTANCE", "ORIGIN", "ORIGIN_STATE_NM", "DEST", "DEST_STATE_NM", 
                                        "NAS_DELAY", "CARRIER_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY","SECURITY_DELAY", 
                                        "*") \
                                .sort("FL_DATE", "CRS_DEP_TIME"))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Cross check tail number in airlines data

# COMMAND ----------

display(df_airlines.where((df_airlines.FL_DATE >= "2019-07-24") &
                        (df_airlines.FL_DATE <= "2019-07-25") &
                              (df_airlines.TAIL_NUM == "N249PS") \
                               ).select("FL_DATE", "CRS_DEP_TIME", 
                                        "TAIL_NUM", "OP_CARRIER_FL_NUM",
                                        "ARR_DEL15", "DISTANCE", "ORIGIN", "ORIGIN_STATE_NM", "DEST", "DEST_STATE_NM", 
                                        "NAS_DELAY", "CARRIER_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY","SECURITY_DELAY", 
                                        "*") \
                                .sort("FL_DATE", "CRS_DEP_TIME"))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Analysis
# MAGIC 
# MAGIC <p>
# MAGIC   
# MAGIC * The model predicted that there would be no delay, but the flight actually got delayed.
# MAGIC * This particular instance has no flight arrival or departure delay in the 2 hour window. 
# MAGIC * Nothing extraordinary stands out in the weather.
# MAGIC * Turns out that the actual arrival flight arrived after the prediction was made and it arrived late. 
# MAGIC * The model did not have sufficient data and predicted a non-delay based on the-then available data.
# MAGIC * Due to unforeseen data, where the flight actually got delayed in arrival from the prior leg of the journey impacted the prediction. 
# MAGIC * This case of False Negative will be hard to guage correctly given the data points that are available when making the prediction.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Investigation #4 False Negative Case

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Pick a record with FN

# COMMAND ----------

display(pred_fn.where((pred_fp.FL_DATETIME_UTC == "2019-06-12T00:38:00.000+0000") & (pred_fp.OP_CARRIER_FL_NUM == 269)))

# COMMAND ----------

display(df_airlines.where((df_airlines.FL_DATE <= "2019-06-12") &
                        (df_airlines.FL_DATE >= "2019-06-11") &
                              (df_airlines.OP_CARRIER_FL_NUM == 269) \
                               ).select("FL_DATE", "CRS_DEP_TIME", 
                                        "TAIL_NUM", "OP_CARRIER_FL_NUM",
                                        "ARR_DEL15", "DISTANCE", "ORIGIN", "ORIGIN_STATE_NM", "DEST", "DEST_STATE_NM", 
                                        "NAS_DELAY", "CARRIER_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY","SECURITY_DELAY", 
                                        "*") \
                                .sort("FL_DATE", "CRS_DEP_TIME"))

# COMMAND ----------

display(df_airlines.where((df_airlines.FL_DATE <= "2019-06-12") &
                        (df_airlines.FL_DATE >= "2019-06-11") &
                              (df_airlines.ORIGIN == "ORD") \
                               ).select("FL_DATE", "CRS_DEP_TIME", 
                                        "TAIL_NUM", "OP_CARRIER_FL_NUM",
                                        "ARR_DEL15", "DISTANCE", "ORIGIN", "ORIGIN_STATE_NM", "DEST", "DEST_STATE_NM", 
                                        "NAS_DELAY", "CARRIER_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY","SECURITY_DELAY", 
                                        "*") \
                                .sort("FL_DATE", "CRS_DEP_TIME"))

# COMMAND ----------

display(df5.where((df2.FL_DATE == "2019-06-11") & \
                   (df2.CRS_DEP_TIME >= 1538) & \
                   (df2.CRS_DEP_TIME <= 1938 )
                 )
       )

# COMMAND ----------

df5 = df_airlines.where((df_airlines.FL_DATE <= "2019-06-12") &
                        (df_airlines.FL_DATE >= "2019-06-11") &
                              (df_airlines.TAIL_NUM == "N416UA") \
                               ).select("FL_DATE", "CRS_DEP_TIME", 
                                        "TAIL_NUM", "OP_CARRIER_FL_NUM",
                                        "ARR_DEL15", "DISTANCE", "ORIGIN", "ORIGIN_STATE_NM", "DEST", "DEST_STATE_NM", 
                                        "NAS_DELAY", "CARRIER_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY","SECURITY_DELAY", 
                                        "*") \
                                .sort("FL_DATE", "CRS_DEP_TIME")
display(df5)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Analysis
# MAGIC 
# MAGIC <p>
# MAGIC   
# MAGIC * In this particular instance, there was no arrival delays but there was 45% delays for the departure at the origin. Despite this, the model did not predict a departure delay.
# MAGIC * The prediction when arrival delay is zero but there is departure delay for the tail number at the time of the prediction.
# MAGIC * The tail number in the hours 15:38 and 17:38 had no delays, however the flight had an arrival delay at 17:47, which is after the time the prediction was made. 
# MAGIC * So again, this is an example of prior legs of the flight being short distance flights, where the model is penalized for its prediction based on limited data.
# MAGIC * To solve this, there is a need to address the case of short distance flights and even consider if the 2 hour window is the appropriate threshold for datapoints across the board.
# MAGIC * Despite the above analysis, the original question of why did the model not predict the delay based on the overall number of delays at origin still remains open. 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Next Steps and further investigation
# MAGIC 
# MAGIC <p>
# MAGIC   
# MAGIC * Evaluate additional records for FP and FN.
# MAGIC * Quantify the patterns observed.
# MAGIC * Evaluate delay networks and graph based models.
# MAGIC * Consider weather forecast as a signal to the model.
# MAGIC * Consider features for extreme weather events and cascading effects due to such conditions on the overall air traffic network.
# MAGIC   