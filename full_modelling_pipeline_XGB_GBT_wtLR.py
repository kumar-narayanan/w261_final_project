# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Imports and environment

# COMMAND ----------

spark.sparkContext.addPyFile("dbfs:/custom_cv.py")

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import max, min, when, col, avg, first
from pyspark.ml.feature import VectorAssembler, VectorSlicer, StandardScaler, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator 
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.ml.tuning import ParamGridBuilder
from custom_cv import CustomCrossValidator
from pyspark.sql.types import FloatType
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import precision_recall_curve

# COMMAND ----------

blob_container = "team08-container" # The name of your container created in https://portal.azure.com
storage_account = "team08" # The name of your Storage account created in https://portal.azure.com
secret_scope = "team08-scope" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "team08-key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Data

# COMMAND ----------

df_airlines_weather_full = spark.read.parquet(f"{blob_url}/df_airlines_weather_holidays")
df_airlines_weather_full.cache()
df_airlines_weather = df_airlines_weather_full.where(df_airlines_weather_full.FL_DATE_UTC < '2019-01-01')
df_airlines_weather.cache()
df_airlines_weather_19 = df_airlines_weather_full.where(df_airlines_weather_full.FL_DATE_UTC >= '2019-01-01')
df_airlines_weather_19.cache()

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Preprocessing

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

# COMMAND ----------

def sample_dataframe(dataframe):
    """
    Helper method to downsample the majority class: non delayed flights
    Input: dataframe to downsample
    Output: downsampled dataframe
    """
    #split into classes
    delayed_df = dataframe.where(dataframe.DEP_DEL15 == 1)
    non_delayed_df = dataframe.where(dataframe.DEP_DEL15 == 0)
    
    #Select rows from each class. The percent values were pre computed to end up with a 50-50 split between classes
    delayed_df_samp = delayed_df.sample(False, 0.90, seed = 2018)
    non_delayed_df_samp = non_delayed_df.sample(False, 0.20, seed = 2018)

    #Combine and return
    return delayed_df_samp.union(non_delayed_df_samp)

# COMMAND ----------

# MAGIC %md
# MAGIC # Modeling Utilities

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

def fold_train_test_assignment_lrw(dataframe, folds, train_weight, features_col = "features", standardize=False, scaled_features_col="scaled_features"):
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
    processed_wt = {}
    
    #Create provided number of folds
    for i in range(folds):
        
        #Determine how many dates in each fold to have even sizing
        cutoff_date = min_date + ((max_date - min_date) / folds) * (i+1)
        
        #If this is the last fold, include the final record used as cutoff point. Otherwise, that record goes in the next fold. 
        if i+1 == folds: 
            cur_fold_df = dataframe.where((dataframe.FL_DATETIME_UTC >= prev_cutoff) & (dataframe.FL_DATETIME_UTC <= cutoff_date))
        else: 
            cur_fold_df = dataframe.where((dataframe.FL_DATETIME_UTC >= prev_cutoff) & (dataframe.FL_DATETIME_UTC < cutoff_date))
        
        ##########################################################################
        count_delay = cur_fold_df[cur_fold_df["DEP_DEL15"] == 1.0].count()
        count_no_delay = cur_fold_df[cur_fold_df["DEP_DEL15"] == 0.0].count()
        #Compute weight by inverse proportion to the occurance of each class
        wt = ((count_delay + count_no_delay) / (count_delay * 2), (count_delay + count_no_delay) / (count_no_delay * 2))
        
        #Add weights to dataframe
        cur_fold_df = cur_fold_df.withColumn("WEIGHT", when(col("DEP_DEL15") == 1.0, wt[0]).otherwise(wt[1]))
        print("Fold:", i+1, "|Weight for delayed flights:", wt[0], "|Weight for non-delayed flights:", wt[1])
        ########################################################################## 
        
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
        processed_wt["df" + str(i+1)] = wt
        
        #Start point for next fold is end point of previous fold
        prev_cutoff = cutoff_date
    
    return processed_df, processed_wt, scaled_train

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


# COMMAND ----------

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
    #ranked_features = ExtractFeatureImp(model.featureImportances, split_df, features_col)
    #print("Ranked features by importance...")
    
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

def fit_optimal_model(classifier, dataframe, feature_count, ranked_features, evaluator, grid, train_weight = 0.8, features_col = "features", folds=5, standardize=False, scaled_features_col="scaled_features", reduce_features=True):
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

def fit_optimal_model_lrw(classifier, dataframe, feature_count, ranked_features, evaluator, grid, train_weight = 0.8, features_col = "features", folds=5, standardize=False, scaled_features_col="scaled_features", reduce_features=True):
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
    dfs, wts, scaler = fold_train_test_assignment_lrw(reduced_df, folds, train_weight, features_col="features_reduced", standardize=standardize, scaled_features_col=scaled_features_col)
    print("Created folds...")
    
    #Set column if standardization used
    if standardize:
        classifier.setFeaturesCol(scaled_features_col)
    
    #Run CustomCrossValidator and return optimal model
    cv = CustomCrossValidator(estimator=classifier, estimatorParamMaps=grid, evaluator=evaluator, splitWord = ('train', 'test'), cvCol = 'cv', parallelism=4)
    return cv.fit(dfs), wts, scaler

# COMMAND ----------

# DBTITLE 1,Super Class of BinaryClassificationMetrics for ROC/PR curves
class CurveMetrics(BinaryClassificationMetrics):
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

    def get_curve(self, method):
        rdd = getattr(self._java_model, method)().toJavaRDD()
        return self._to_list(rdd)

# COMMAND ----------

def test_model(model, dataframe, evaluator, reduce_features=True, feature_count=None, ranked_features=None, label_col='DEP_DEL15', features_col='features', scaler=None, scaled_features_col='scaled_features'):
    """
    Helper method to test model and output evaluation metrics
    Input: Model, dataframe to test on, evaluator to use, boolean to indicate feature reduction, number of features to use, ranked list of features, label column, features column, scaler to use, location of scaled features
    Output: returns predictions. Prints AUC, ROC, Confusion Matrix and PR curve
    """
    
    #Reduce to requested number of features if required
    if reduce_features:
        features_to_use = ranked_features[0:feature_count]
        feature_idx = [x for x in features_to_use['idx']]
        slicer = VectorSlicer(inputCol=features_col, outputCol="features_reduced", indices=feature_idx)
        reduced_df = slicer.transform(dataframe)
    else:
        #Change column name to keep consistent 
        reduced_df = dataframe.withColumn("features_reduced", col(features_col))

    #Scale features if required
    if scaler:
        print("Scaling features...")
        reduced_df = scaler.transform(reduced_df)
    
    #Run test
    predictions = model.transform(reduced_df)

    #Output AUC
    print("AUC: " + str(evaluator.evaluate(predictions)))

    #Select probabilities to plot ROC
    preds_for_plot = predictions.select(label_col,'probability').rdd.map(lambda row: (float(row['probability'][1]), float(row[label_col])))
    points = CurveMetrics(preds_for_plot).get_curve('roc')

    #Generate plots
    fig, axs = plt.subplots(nrows=3, figsize=(20,20))

    #Plot ROC
    x_val = [x[0] for x in points]
    y_val = [x[1] for x in points]
    axs[0].set_title("ROC")
    axs[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate")
    axs[0].plot(x_val, y_val, )
    
    #Select predictions and labels for Confusion Matrix
    preds_and_labels = predictions.select(['prediction','DEP_DEL15']).withColumn('label', col('DEP_DEL15').cast(FloatType())).orderBy('prediction').select(['prediction','label'])

    #Generate CM
    metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
    cm = metrics.confusionMatrix().toArray()

    #Label CM
    df_cm = pd.DataFrame(cm, index = ["True Not Delayed", "True Delayed"],
                  columns = ["Predicted Not Delayed", "Predicted Delayed"])

    #Plot CM
    sn.heatmap(df_cm, annot=True, ax=axs[1], fmt='.0f', vmin=100000, vmax=3000000)
    
    #Plot PR Curve
    pr_points = CurveMetrics(preds_for_plot).get_curve('pr')
    axs[2].set_title("PR")
    axs[2].set(xlabel="Recall", ylabel="Precision")
    axs[2].plot([x[0] for x in pr_points], [x[1] for x in pr_points], )
    
    return predictions

# COMMAND ----------

# MAGIC %md
# MAGIC # Models

# COMMAND ----------

# MAGIC %md
# MAGIC ## XGB

# COMMAND ----------

#prepare data
preprocessed_df_xgb = sample_dataframe(prepare_dataframe_full(simplify_dataframe_full(df_airlines_weather)[0])[0])
preprocessed_df_xgb.cache()

# COMMAND ----------

#fit model
from sparkdl.xgboost import XgboostClassifier

xgb = XgboostClassifier(labelCol="DEP_DEL15", featuresCol="features", missing=0.0)
evaluator = BinaryClassificationEvaluator(labelCol="DEP_DEL15")
grid = ParamGridBuilder()\
        .addGrid(xgb.max_depth, [5, 10]).build()

#feature_count, ranked_features = determine_optimal_feature_count(xgb, preprocessed_df_xgb, feature_counts, evaluator)
model_xgb, scaler = fit_optimal_model(xgb, preprocessed_df_xgb, None, None, evaluator, grid, reduce_features=False)
#display(ranked_features)

# COMMAND ----------

#test model
preprocessed_df_19_xgb = prepare_dataframe_full(simplify_dataframe_full(df_airlines_weather_19)[0])[0]
preprocessed_df_19_xgb.cache()
predictions_xgb = test_model(model_xgb, preprocessed_df_19_xgb, evaluator, scaler=scaler, reduce_features=False)
# display(predictions_rf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## GBT

# COMMAND ----------

#prepare data
df, encoder = simplify_dataframe_full(df_airlines_weather, ohe=True)
preprocessed_df_gbt = sample_dataframe(prepare_dataframe_full(df)[0])
preprocessed_df_gbt.cache()

# COMMAND ----------

#fit model
from pyspark.ml.classification import GBTClassifier
gbt = GBTClassifier(labelCol="DEP_DEL15", featuresCol="features")
evaluator = BinaryClassificationEvaluator(labelCol="DEP_DEL15")
grid = ParamGridBuilder() \
               .addGrid(gbt.maxIter, [5, 10])\
               .addGrid(gbt.maxDepth, [5, 10])\
               .build()

evaluator = BinaryClassificationEvaluator(labelCol="DEP_DEL15")

model_gbt, scaler = fit_optimal_model(gbt, preprocessed_df_gbt, None, None, evaluator, grid, reduce_features=False)

# COMMAND ----------

#Test model
preprocessed_df_19_gbt = prepare_dataframe_full(simplify_dataframe_full(df_airlines_weather_19, ohe=True, encoder=encoder)[0])[0]
preprocessed_df_19_gbt.cache()
predictions_gbt = test_model(model_gbt, preprocessed_df_19_gbt, evaluator, scaler=scaler, reduce_features=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Weighted Logistic Regression

# COMMAND ----------

df, encoder = simplify_dataframe_full(df_airlines_weather, ohe=True)
preprocessed_df_lrw, wt = prepare_dataframe_full(df)
preprocessed_df_lrw.cache()

# COMMAND ----------

#fit model
lrw = LogisticRegression(labelCol="DEP_DEL15", 
                                featuresCol="scaled_features", 
                                standardization=False,
                                threshold=0.5, 
                                weightCol="WEIGHT")
evaluator = BinaryClassificationEvaluator(labelCol="DEP_DEL15")
grid = ParamGridBuilder()\
        .addGrid(lrw.elasticNetParam, [0, 0.5, 1])\
        .addGrid(lrw.aggregationDepth, [4, 8])\
        .build()

model_lrw, wts, scaler = fit_optimal_model_lrw(lrw, preprocessed_df_lrw, None, None, evaluator, grid, standardize=True, scaled_features_col="scaled_features", reduce_features=False)

# COMMAND ----------

wt_delay, wt_no_delay = 0, 0
for _, val in wts.items():
    wt_delay += val[0]
    wt_no_delay += val[1]

wt_delay = wt_delay / len(wts)
wt_no_delay = wt_no_delay / len(wts)

wt = (wt_delay, wt_no_delay)
print("Weight for delayed flights:", wt_delay, "|Weight for non delayed flights:", wt_no_delay)

# COMMAND ----------

preprocessed_df_19_lrw = prepare_dataframe_full(simplify_dataframe_full(df_airlines_weather_19, ohe=True, encoder=encoder)[0], include_wt=True, wt=wt)[0]
preprocessed_df_19_lrw.cache()
predictions_lrw = test_model(model_lrw, preprocessed_df_19_lrw, evaluator, reduce_features=False, scaler=scaler)

# COMMAND ----------

# MAGIC %md
# MAGIC # Save Models

# COMMAND ----------

model_xgb.write().save(f"{blob_url}/models/xgb_optimized")

# COMMAND ----------

model_gbt.write().save(f"{blob_url}/models/gbt_optimized")


# COMMAND ----------

model_lrw.write().overwrite().save(f"{blob_url}/models/lrw_optimized")

# COMMAND ----------

# MAGIC %md
# MAGIC # Save Predictions

# COMMAND ----------

predictions_xgb.write.mode("overwrite").parquet(f"{blob_url}/predictions/xgb_optimized")

# COMMAND ----------

predictions_gbt.write.mode("overwrite").parquet(f"{blob_url}/predictions/gbt_optimized")

# COMMAND ----------

predictions_lrw.write.mode("overwrite").parquet(f"{blob_url}/predictions/lrw_optimized")