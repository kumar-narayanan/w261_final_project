# Databricks notebook source
# DBTITLE 1,Imports 
from pyspark.sql.functions import max, min, when, col, avg, first, lit, struct, udf
from pyspark.ml.evaluation import BinaryClassificationEvaluator 
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql.types import IntegerType, StringType
import matplotlib.pyplot as plt
import random

# COMMAND ----------

# DBTITLE 1,Set up Environment
blob_container = "team08-container" # The name of your container created in https://portal.azure.com
storage_account = "team08" # The name of your Storage account created in https://portal.azure.com
secret_scope = "team08-scope" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "team08-key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# COMMAND ----------

# DBTITLE 1,Read the data and separate 2019 data
df_airlines_weather_full = spark.read.parquet(f"{blob_url}/df_airlines_weather_holidays")
df_airlines_weather_full.cache()
df_airlines_weather = df_airlines_weather_full.where(df_airlines_weather_full.FL_DATE_UTC < '2019-01-01')
df_airlines_weather.cache()
df_airlines_weather_19 = df_airlines_weather_full.where(df_airlines_weather_full.FL_DATE_UTC >= '2019-01-01')
df_airlines_weather_19.cache()

# COMMAND ----------

# DBTITLE 1,Get relative probabilities of delay and no_delay
count_delay = df_airlines_weather[df_airlines_weather["DEP_DEL15"] == 1.0].count()
count_no_delay = df_airlines_weather[df_airlines_weather["DEP_DEL15"] == 0.0].count()

p_delay = count_delay/(count_delay + count_no_delay)
p_no_delay = count_no_delay/(count_delay + count_no_delay)

print("Delay probability:", p_delay, "| No delay probability:", p_no_delay)

# COMMAND ----------

# DBTITLE 1,Class definition to obtain ROC and PR curves
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

# DBTITLE 1,Function to provide 1 or 0 according to probability distribution
def flip():
    global p_delay
    return 1 if random.random() < p_delay else 0

flip_udf = udf(flip, IntegerType())

# COMMAND ----------

# DBTITLE 1,Null model assuming that the flights are never delayed
df_airlines_weather_19_out = df_airlines_weather_19.withColumn("predictions", lit(0)).withColumn("rawPrediction", lit(0.0)).withColumn("probability", struct(lit(1.0), lit(0.0)))
display(df_airlines_weather_19_out)

# COMMAND ----------

# DBTITLE 1,Evaluate the predicted frame
evaluator = BinaryClassificationEvaluator(labelCol="DEP_DEL15")
evaluator.evaluate(df_airlines_weather_19_out)

# COMMAND ----------

# DBTITLE 1,Plot the ROC curve
preds_for_plot = df_airlines_weather_19_out.select("DEP_DEL15","probability").rdd.map(lambda row: (float(row['probability'][1]), float(row["DEP_DEL15"])))
points = CurveMetrics(preds_for_plot).get_curve('roc')

plt.figure()
x_val = [x[0] for x in points]
y_val = [x[1] for x in points]
plt.title("ROC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.plot(x_val, y_val)

# COMMAND ----------

# DBTITLE 1,Data frame knowing the probabilities of delay and no_delay
df_airlines_weather_19_out = df_airlines_weather_19.withColumn("predictions", flip_udf()).withColumn("rawPrediction", col("predictions")* 1.0).withColumn("probability", struct(lit(p_delay), lit(p_no_delay)))
display(df_airlines_weather_19_out)

# COMMAND ----------

# DBTITLE 1,Evaluate the predicted frame
evaluator = BinaryClassificationEvaluator(labelCol="DEP_DEL15")
evaluator.evaluate(df_airlines_weather_19_out)

# COMMAND ----------

# DBTITLE 1,Plot the ROC curve
preds_for_plot = df_airlines_weather_19_out.select("DEP_DEL15","probability").rdd.map(lambda row: (float(row['probability'][1]), float(row["DEP_DEL15"])))
points = CurveMetrics(preds_for_plot).get_curve('roc')

plt.figure()
x_val = [x[0] for x in points]
y_val = [x[1] for x in points]
plt.title("ROC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.plot(x_val, y_val)

# COMMAND ----------

