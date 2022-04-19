# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Predicting Flight Delays Based on Weather
# MAGIC 
# MAGIC ## Team 08 -- W261 Final Project
# MAGIC 
# MAGIC ### Mrinal Chawla, Kumar Narayanan, Sushant Joshi, Ajeya Tarikere Jayaram

# COMMAND ----------

# MAGIC %md
# MAGIC #Phases Summaries
# MAGIC 
# MAGIC ## Phase 1
# MAGIC 
# MAGIC Phase 1 involves getting familiar with the question asked and the data available. EDA was performed with the intention of determining the output variable, how to measure and compare models, what features are likely to be useful and how to join the available datasets. This investigation also included anticipating potential challenges.
# MAGIC 
# MAGIC From the above investigation, it was found that the DEP_DEL15 feature is the best output variable to track against with f1-score as the best measure to use in model comparison. On the feature side, it was found that the majority of delays were due to weather. Therefore, the weather features will be important in model formation. This brought up the challenge of which weather data to consider. When conducting the join of the weather and flight datasets, we will need to determine the closest weather station to the relevant points in the flight path, and join the relevant weather data prior to departure to each flight record. 
# MAGIC 
# MAGIC ## Phase 2
# MAGIC 
# MAGIC During Phase 2, we continued the EDA on the airlines and weather data. We worked on selecting simple features from the airlines and the weather data for the baseline model. We successfully joined the weather and airlines data to get weather 2 hours prior to a flight's departure time for 3 months and 6 months datasets. We then ran logistic regression and decision tree baseline models on the datasets and obtained a F1 score of 0.5516 on 3 months data and F1 score of .4973 on 6 months data using decision tree. The logistic regression, on the other hands, gave an F1 score of 0.1766 for the 3-months data. The F1 score with the 6-months data is 0.0. This turned out to be due to zero True Positive (TP) reported by the logistic regression. The  raw data without feature engineering, normalization etc. isn't a good model for Logistic Regression. 
# MAGIC 
# MAGIC ## Phase 3
# MAGIC 
# MAGIC In this phase, we evaluated features from the weather and airlines dataset, selected a subset of those features, performed data cleaning and feature engineering on those features. The other milestone that we reached successfully was joining the full weather and airlines data sets. It took roughly 7 hours to perform the join. Due to the imbalanced data set, we attempted the SMOTE algorithm for oversampling the minority class, but so far we did not find any major benefit on the results yet, so we under-sampled the majority class. For the models, we ran Logistic Regression and Decision Tree on the full data set with the additional features included. The results of the Logistic Regression is a F1 Score of 0.34, and that for the Decision Tree is 0.32. 

# COMMAND ----------

# MAGIC %md
# MAGIC # Question Formulation
# MAGIC 
# MAGIC ## Business Use Case
# MAGIC 
# MAGIC Predicting flight delays is a lucrative opportunity for airports and airlines alike. For a flight passenger, knowing the exact time of departure can help in planning transport to and from airports for pick up and drop off, itenraries for travel and provide ease of mind in a already stressful situation. If an airport or airline can consistently provide delay information to passengers, they are likely to see customer trust develop and a corresponding increase in revenue. Flight delays have a ripple effect and can cause disruptions to air transportation schedules. Delayed flights have a negative economic impact on the airlines operator, the air travelers, and on the airports. In the US, the impact of domestic flight delays was estimated to be $30-40 billion annually. By being able to predict flights, air traffic control and airlines could put contingency measures in place to prevent this ripple effect and minimize the economic impact of flight delays.
# MAGIC 
# MAGIC ## Research Question
# MAGIC 
# MAGIC This problem is a classification problem, containing 2 possible classes, delayed or not delayed. Delayed will be defined as departure being 15 or more minutes after scheduled departure time. This determination needs to be made 2 hours before the scheduled departure, in order to provide benefits to the customers as discussed in the Business Use Case. Consisely, the question can be formulated as: 
# MAGIC 
# MAGIC _Will a flight's departure be delayed by 15 or more minutes, based on weather and airport conditions 2 hours prior to scheduled departure?_
# MAGIC 
# MAGIC _Note_: As an additional step we could consider a regression model to predict the actual delay (beyond saying if the delay is 15 min. or less) and establish scores for a few buckets. For instance, we could consider the accuracy scores for delays in the 15- to 30- minutes category, 30- to 45- minutes category etc.
# MAGIC 
# MAGIC ## Evaluation 
# MAGIC 
# MAGIC __Terminology__
# MAGIC 
# MAGIC Positve: The flight is delayed by 15 minutes. 
# MAGIC 
# MAGIC Negative: The flight is not delayed by 15 minutes
# MAGIC 
# MAGIC True Positive: The flight was delayed and the model predicted it was delayed. 
# MAGIC 
# MAGIC False Positive: The flight was not delayed and the model predicted it was delayed. 
# MAGIC 
# MAGIC True Negative: The flight was not delayed and the model predicted it was not delayed
# MAGIC 
# MAGIC False Negative: The flight was delayed and the model predicted it was not delayed. 
# MAGIC 
# MAGIC __Customer Experience__
# MAGIC 
# MAGIC True Positive: This is a good experience. The customer was correctly notified that the flight was delayed. 
# MAGIC 
# MAGIC False Positive: This is a bad experience. The customer expected a delay, but the flight was on time. this could mean missed flights or extra effort spent planning for delay
# MAGIC 
# MAGIC True Negative: This is a neutral experience. The customer was not notified of any delay and there was no delay. 
# MAGIC 
# MAGIC False Negative: This is a bad experience. The customer was not notified of any delay but there was a delay. This experience can also be considered a status quo, since this would be the experience if there was no model / prediction made. 
# MAGIC 
# MAGIC __The Metric__
# MAGIC 
# MAGIC Based on the customer experiences above, both a False Positive and a False Negative are bad experiences. Therefore, the metric that will be used to evaluate the models will be the f1-score, that focuses on driving down both False predictions. While accuracy, would have been the most straightforward metric, it is easy to get a high accuracy score by predicting the majority class, since the data is skewed away from delayed flights. Therefore f1-score is the chosen metric. 
# MAGIC 
# MAGIC $$f_1 = 2 * \frac{TP}{TP+\frac{1}{2}(FP+FN)}$$
# MAGIC 
# MAGIC 
# MAGIC Another metric to track is a F-score with a beta parameter of 0.5. This will double the penality for a False Positive over a False Negative. It may make sense to have this formula, since a False Positive is a worse experience than a False Negative. This metric will also be tracked as a secondary evaluation point. 
# MAGIC 
# MAGIC $$f_{0.5} = 1.25 * \frac{1.25 * TP}{1.25 * TP + 0.25 * FN + FP}$$

# COMMAND ----------

# MAGIC %md
# MAGIC # EDA & Discussion of Challenges
# MAGIC 
# MAGIC <br>
# MAGIC 
# MAGIC ## Preliminary EDA
# MAGIC 
# MAGIC ### [Weather Dataset](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1898361324233996/command/1898361324233997)
# MAGIC 
# MAGIC | Description | Notes |
# MAGIC |---|---|
# MAGIC | Number of Datapoints in full Data Set |  630,904,436 | 
# MAGIC | Number of Datapoints in 1st Quarter of 2015 | 29,823,926 |
# MAGIC | Total Number of Features in Raw | 177 |
# MAGIC | Total Number of Retained Features | 20* |
# MAGIC | Interesting Feature Groups | Air Temperature, Dew Point Temperature, Windspeed, Visibility, Sea level Pressure, Precipitation, Snow, Fog, Thunder, Gust among other available parameters |
# MAGIC | Required Features | Date, Latitude, Longitude, Station |
# MAGIC | Dropped Features | Quality Code Features, Features from Additional Section |
# MAGIC 
# MAGIC *(*Evaluating the data quality of Precipitation, Snow, Fog, Thunder, Gust among other available parameters)*
# MAGIC 
# MAGIC ### [Flights Dataset](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1898361324231581/command/1898361324231584)
# MAGIC | Description | Notes |
# MAGIC |---|---|
# MAGIC | Total Number of DataPoints in Full Data Set | 31,746,841 |
# MAGIC | Total Number of DataPoints in 1st Quarter of 2015 | 161057 |
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
# MAGIC 
# MAGIC <br>
# MAGIC 
# MAGIC ### [Joins](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1898361324236910/command/1898361324236913)
# MAGIC 
# MAGIC Most often, flights are delayed due to inclement weather conditions. Hence, knowing the weather conditions around the time of the flight would be crucial to predict flight delays. As the airline operators and the airport staff need sufficient time to take measures to address the flight delays, we will need to predict weather delays at least 2 hours prior to a flight's departure time.
# MAGIC We have three datasets available with us - 
# MAGIC 1. the **airlines** data, which contains information about individual flights such as flight number, flight date, carrier, departure time, origin airport, destination airport.
# MAGIC 1. the **weather** data, which contains weather measurements such as wind speed, temperature, dew point, station number, tinestamp from different weather stations, 
# MAGIC 1. and the **stations** data, which has the latitude and longitude of the station and its nearest neighbors and the distace to the nearest neighbors. 
# MAGIC 
# MAGIC The airlines data has the codes for the origin and destination airports, but no location information about these airports. We will employ a **secondary airport data** that provides the location (latitude and longitude) information of these airports. 
# MAGIC In order to get the weather condition at the airport 2 hours prior to departure time we will:
# MAGIC 1. **filter out only stations which have themselves as the neighbors** since the stations table contains the distance of a station to all neighboring stations along with distance to itself.
# MAGIC 1. Create a lookup table, **airport_station_lookup**, by cross joining the airport codes and unique station, calculating the **Haversine distance** between the airports' latitude and longitude and the stations' latitude and longitude and filtering only the stations which have the smallest distance to the airport.
# MAGIC 1. Next, we will apply a left join on airlines table and the airport_stations_lookup table on the **airport code** (of the origin or destination airports, depending on whether we need weather at the origin or at the destination).
# MAGIC 1. Since the airlines table has departure time in local or origin airport timezone, whereas the weather table has date in UTC, the flight departure time would be converted to UTC to facilitate joins.
# MAGIC 1. Next, left join these with the weather table on the **station** field where the **date** timestamp is less than 2 hours of the **departure time of the flight**.
# MAGIC 1. Finally, we will compute the average weather condition using a 2 hour window at the airport measured 2 hours prior to the departure time of the flight for each of these flight, i.e. if the flight departs at 10 AM, we will use weather data at the airport from 6AM to 8AM to make the flight delay predictions.
# MAGIC 
# MAGIC We use a left join on the airlines table as we need to use all the flight data even if we do not have the weather conditions
# MAGIC 
# MAGIC <br>
# MAGIC 
# MAGIC ### Feature Selection
# MAGIC 
# MAGIC There are many fields that are available in the flights dataset. Through initial analysis, it was found that weather is the biggest factor in flight delays. There are also issues with the database, such as missing values, and columns with multiple features included, that need to be sorted before training any model. Finally, not all features will be relevant to prediction. Based on availability and charatersitics, some features will need to be removed before training. 
# MAGIC 
# MAGIC Since we cannot use any of the departure performance features, flight summaries features, cause of delay features for the prediction and since some of the features such as carrier features, origin or destination airport features have redundant features which do not add value to prediction, our choice of features to use would be -
# MAGIC Flight Number, Airline, Origin Airport, Destination Airport, Flight Departure Time and the target variable DEP_DEL15.  All other prediction variables would come from weather table.
# MAGIC 
# MAGIC <br>
# MAGIC 
# MAGIC ### Data Representation
# MAGIC 
# MAGIC With a large dataset, the data needs to be represented efficiently in the cluster. It was found that the best representation would be a column based representation, such as parquet. Column based storage stores all the values for a particular column close together in memory. This allows queries with filters to run much more efficiently, as only a few bytes for the column have to be read, instead of the entire row, as with a row based storage solution. Furthermore, if the filtering can be processed with only a few bytes, the entire row can be read post filtering, which saves much more time in reading. This is an ideal solution for this type of data where there are many columns in each data set, particularily once the join is done. 
# MAGIC 
# MAGIC <br>
# MAGIC 
# MAGIC ### Train / Test Split
# MAGIC 
# MAGIC This dataset will need to be split in a way to prevent data leakage. This is to prevent previous information about a flight, such as flight ids, airplane codes, etc. in the train data being used to predict a fligth delay rather than features such as weather and airport conditions being used. Therefore, a split will have to be taken with the time of each record considered. This is to ensure that the test data is also always after the train data, since knowing later data in the training stage will ruin predictions for previous flights. One strategy to handle this is to split based on day, and assign each day to train or test set. This will ensure no data leakage, and that the model will primarily use the intended features in prediction. 

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC #Algorithm Exploration
# MAGIC 
# MAGIC For the initial baseline, a Decision Tree Model and a Logistic Regression model were used. 
# MAGIC These are both models that fit with a binary classification problem. Logistic Regression typically provies a strong classifier for most use cases. Decision Trees are robust to outliers and skewed data. Since there is minimal feature engineering done at this stage, these models will provide the baseline to improve upon and track success. 

# COMMAND ----------

# MAGIC %md
# MAGIC #Algorithm Implementation

# COMMAND ----------

def simplify_dataframe(dataframe):
    """
    Helper method to select relevant features and prepare for running through a model
    Input: dataframe containing all data after joining
    Output: dataframe containing only features for use in model
    """
    #Drop rows where output is missing
    df = df.where((df.DEP_DEL15).isNotNull())
    
    #Select features and drop columns used for join
    relevant_columns = ["FL_DATE_UTC", "DEP_HOUR_UTC", "OP_CARRIER_FL_NUM", "ORIGIN", "DEST", "ELEVATION", "WIND_GUST_SPEED_RATE", "WIND_DIR_IMP", "WIND_SPEED_RATE_IMP", "VIS_DISTANCE_IMP", "CEILING_HEIGHT_IMP", "AIR_TEMP_IMP", "DEW_POINT_TEMPERATURE_IMP", "SEA_LEVEL_PRESSURE_IMP", "ATMOS_PRESSURE_3H_IMP", "PRECIPITATION_RATE", "PRESENT_ACC_OHE", "PAST_ACC_OHE", "ATMOS_PRESSURE_TENDENCY_3H_OHE", "WIND_TYPE_OHE", "DEP_DEL15"]

    #Aggregate weather records in the window
    simplified_dataframe = df.select([col for col in relevant_columns])
    simplified_dataframe = simplified_dataframe.groupBy(simplified_dataframe.FL_DATE_UTC, simplified_dataframe.DEP_HOUR_UTC, simplified_dataframe.OP_CARRIER_FL_NUM, simplified_dataframe.ORIGIN) \
                                   .agg(max("DEST").alias("DEST"), \
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
    return simplified_dataframe

def prepare_dataframe(dataframe):
    """
    Helper method to prepare dataframe for use in MLLib models
    Input: Dataframe after feature engg and aggregation is done
    Output: Dataframe containing "features" column to use in model
    """
    #Drop records with missing weather data
    non_null_df = df.where((df.ELEVATION).isNotNull())
    
    #Select features to assemble
    feature_columns = ["ELEVATION", "WIND_GUST_SPEED_RATE", "WIND_DIR_IMP", "WIND_SPEED_RATE_IMP", "VIS_DISTANCE_IMP", "CEILING_HEIGHT_IMP", "AIR_TEMP_IMP", "DEW_POINT_TEMPERATURE_IMP", "SEA_LEVEL_PRESSURE_IMP", "ATMOS_PRESSURE_3H_IMP", "PRECIPITATION_RATE", "PRESENT_ACC_OHE", "PAST_ACC_OHE", "ATMOS_PRESSURE_TENDENCY_3H_OHE", "WIND_TYPE_OHE"]

    #Assemle into feature vector
    assembler = VectorAssembler(inputCols = feature_columns, outputCol = "features")
    non_null_df = assembler.transform(non_null_df)

    return non_null_df

def train_test_split(dataframe, train_weight, test_weight, seed = 2018):
    """
    Helper method to split dataframe into train / test based on date
    Input: full dataframe, % of dates for train, % of dates for test, random seed
    Output: train dataframe, test dataframe
    """
    #get all dates present in dataframe
    dates = dataframe.select(dataframe.FL_DATE_UTC).distinct().orderBy(dataframe.FL_DATE_UTC)
    
    #Split dates into train / test
    train_dates, test_dates = dates.randomSplit([train_weight, test_weight], seed)
    train_dates = set(train_dates.toPandas()['FL_DATE_UTC'])
    test_dates = set(test_dates.toPandas()['FL_DATE_UTC'])
    
    #Select records with train / test dates
    train_df = dataframe.where(dataframe.FL_DATE_UTC.isin(train_dates))
    test_df = dataframe.where(dataframe.FL_DATE_UTC.isin(test_dates))
    
    return train_df, test_df

def sample_dataframe(dataframe): 
    """
    Helper method to downsample majority class
    Input: full dataframe
    Output: new dataframe after downsampling
    """
    #split by class
    delayed_df = dataframe.where(dataframe.DEP_DEL15 == 1)
    non_delayed_df = dataframe.where(dataframe.DEP_DEL15 == 0)
    
    #Sample dataframes. Percentage chosen to end with 50-50 split
    delayed_df_samp = delayed_df.sample(False, 0.90, seed = 2018)
    non_delayed_df_samp = non_delayed_df.sample(False, 0.20, seed = 2018)

    return delayed_df_samp.union(non_delayed_df_samp)

def calculate_f1_score(dataframe):
    """
    Helper method to calculate f1 score
    Input: dataframe containing records: (true values, predicted values)
    Output: f1 score
    """
    TP = dataframe.where((dataframe.DEP_DEL15 == 1) & (dataframe.prediction == 1)).count()
    FP = dataframe.where((dataframe.DEP_DEL15 == 0) & (dataframe.prediction == 1)).count()
    TN = dataframe.where((dataframe.DEP_DEL15 == 0) & (dataframe.prediction == 0)).count()
    FN = dataframe.where((dataframe.DEP_DEL15 == 1) & (dataframe.prediction == 0)).count()
    f1 = np.divide(TP, (TP + 0.5*(FP+FN)) )
    return f1

# COMMAND ----------

"""
Decision Tree Algorithm
"""

def N_fold_Validation(N, dataframe, train_weight, test_weight):
    """
    Helper method to run DT Algorithm N times and average results
    Input: number of runs to average, full datframe, train percentage, test percentage
    Output: Average f1 score
    """
    scores = []
    
    #generate N random seeds
    random.seed(2018)
    seeds = random.sample(range(1, 3000), N)
    
    #Iterate N times
    for i in range(N):
        seed = seeds[i]
        
        #Split dataframe with current seed
        train_df, test_df = train_test_split(dataframe, train_weight, test_weight, seed)
        
        #train DT
        dt = DecisionTreeClassifier(labelCol="DEP_DEL15", featuresCol="features", maxDepth=25, minInstancesPerNode=30, impurity="gini")
        pipeline = Pipeline(stages=[dt])
        model = pipeline.fit(train_df)
        
        #Predict 
        predictions = model.transform(test_df)
        pred_labels = predictions.select(predictions.DEP_DEL15, predictions.prediction)

        #Evaluate
        f1 = calculate_f1_score(pred_labels)
        print("Fold " + str(i+1) + ": " + str(f1))
        scores.append(f1)
        
    #Return average    
    return np.mean(scores)

#Prepare and evaluate 3 month dataframe
simplified_dataframe_3m = simplify_dataframe(df_weather_airline_3m)
prepared_dataframe_3m = prepare_dataframe(simplified_dataframe_3m)
f1_score_3m = N_fold_Validation(5, prepared_dataframe_3m, 0.8, 0.2)
print("3-month F1 Score: " + str(f1_score_3m)) #0.5516327569133556

#Prepare and evaluate 6 month dataframe
simplified_dataframe_6m = simplify_dataframe(df_weather_airline_6m)
prepared_dataframe_6m = prepare_dataframe(simplified_dataframe_6m)
f1_score_6m = N_fold_Validation(5, prepared_dataframe_6m, 0.8, 0.2)
print("6-month F1 Score: " + str(f1_score_6m))

# COMMAND ----------

"""
Full dataset evaluation
"""
#Read full joined dataset. Split out 2019 data for testing
df_airlines_weather_full = spark.read.parquet(f"{blob_url}/df_airlines_weather")
df_airlines_weather_full.cache()
df_airlines_weather = df_airlines_weather_full.where(df_airlines_weather_full.FL_DATE_UTC < '2019-01-01')
df_airlines_weather.cache()
df_airlines_weather_19 = df_airlines_weather_full.where(df_airlines_weather_full.FL_DATE_UTC >= '2019-01-01')
df_airlines_weather_19.cache()

#preprocessing for dataframe
s_dataframe = simplify_dataframe_full(df_airlines_weather)
s_dataframe.cache()
p_dataframe = prepare_dataframe_full(s_dataframe)
p_dataframe.cache()
samp_dataframe = sample_dataframe(p_dataframe)
samp_dataframe.cache()

#train/test split
train_df, test_df = train_test_split(samp_dataframe, 0.8, 0.2, 2018)
train_df.cache()
test_df.cache()

#train model
dt = DecisionTreeClassifier(labelCol="DEP_DEL15", featuresCol="features", maxDepth=29, minInstancesPerNode=50, impurity="gini", seed=2018)
model = dt.fit(train_df)
predictions = model.transform(test_df)
predictions.cache()

#preprocessing for test dataframe
s_dataframe_19 = simplify_dataframe_full(df_airlines_weather_19)
s_dataframe_19.cache()
p_dataframe_19 = prepare_dataframe_full(s_dataframe_19)
p_dataframe_19.cache()

#predict
predictions_19 = model.transform(p_dataframe_19)
predictions_19.cache()

#evaluate
print(calculate_f1_score(predictions_19))

# COMMAND ----------

"""
Logistic Regression Algorithm
"""
def logreg(N, dataframe, train_weight, test_weight):
    """
    Helper method to run LR Algorithm N times and average results
    Input: number of runs to average, full datframe, train percentage, test percentage
    Output: Average f1 score
    """
    scores = []
    
    #Generate N random seeds
    random.seed(3000)
    seeds = random.sample(range(1000, 6000), N)
    
    #Iterate N Times
    for i in range(N):
        seed = seeds[i]
        
        #split into train / test
        train_df, test_df = train_test_split(dataframe, train_weight, test_weight, seed)

        #Fit LR model
        lr = LogisticRegression(labelCol="DEP_DEL15", featuresCol="features", maxIter=100, aggregationDepth=8)
        pipeline = Pipeline(stages=[lr])
        model = pipeline.fit(train_df)
        
        #Predict
        predictions = model.transform(test_df)
        pred_labels = predictions.select(predictions.DEP_DEL15, predictions.prediction)
        
        #Evaluate
        f1 = calculate_f1_score(pred_labels)
        print("Iteration " + str(i+1) + ": " + str(f1))
        scores.append(f1)
    
    #Return average
    return np.mean(scores)

#Prepare and evaluate 3 month dataset
simplified_dataframe_3m = simplify_dataframe(df_weather_airline_3m)
prepared_dataframe_3m = prepare_dataframe(simplified_dataframe_3m)
f1_score_3m = logreg(10, prepared_dataframe_3m, 0.725, 0.275)
print("3-month F1 Score: " + str(f1_score_3m)) #0.17664215585712034

#Prepare and evaluate 6 month dataset
simplified_dataframe_6m = simplify_dataframe(df_weather_airline_6m)
prepared_dataframe_6m = prepare_dataframe(simplified_dataframe_6m)
f1_score_6m = logreg(5, prepared_dataframe_6m, 0.75, 0.25)
print("6-month F1 Score: " + str(f1_score_6m)) #0.0

# COMMAND ----------

# MAGIC %md
# MAGIC [Decision Tree Execution](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/4070574709967407/command/4070574709967566)
# MAGIC 
# MAGIC [Logistic Regression Execution](https://adb-731998097721284.4.azuredatabricks.net/?o=731998097721284#notebook/1898361324252165/command/4070574709968194)

# COMMAND ----------

# MAGIC %md
# MAGIC #Conclusions
# MAGIC 
# MAGIC The baseline models for DT are performing better than the models for Logistic Regression. This is to be expected with very minimal feature engineering, as the Decision Trees are more robust to the outliers and heavy skew present in the dataset. There are many more flights that are not delayed than are delayed, leading to the 6 month LR model only prediciting not delayed and having a f1 score of 0. As the feature engineering is undertaken, the skew and the outliers will need ot be addressed. Both models should see marked improvement. 

# COMMAND ----------

# MAGIC %md
# MAGIC #Application-of-Course Concepts

# COMMAND ----------

# MAGIC %md
# MAGIC #References
# MAGIC 
# MAGIC * https://transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr			
# MAGIC * https://www.bts.dot.gov/explore-topics-and-geography/topics/airline-time-performance-and-causes-flight-delays			
# MAGIC * https://www.bts.gov/topics/airlines-and-airports/understanding-reporting-causes-flight-delays-and-cancellations			
# MAGIC * https://www.transtats.bts.gov/printglossary.asp			