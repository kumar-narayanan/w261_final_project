# Databricks notebook source
df_airlines = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data_3m/")
display(df_airlines)

# COMMAND ----------

for col in df_airlines.dtypes:
    print(col[0] + " | " + col[1])

# COMMAND ----------

from pyspark.sql.functions import col
delays_df = df_airlines.filter(df_airlines.DEP_DEL15 > 0)
delays_df.select(col("DEP_DEL15"), col("DEP_DELAY"), col("DEP_DELAY_NEW") ).show()

# COMMAND ----------

print(delays_df.count() )
print(df_airlines.count())
delays_df.count() / float(df_airlines.count())

# COMMAND ----------

null_delays = df_airlines.filter(col("DEP_DEL15").isNull())
null_delays.select("Cancelled").show()
print(null_delays.count())
print(null_delays.select("Cancelled").groupBy().sum().collect())

# COMMAND ----------

# MAGIC %md
# MAGIC __Definitions__
# MAGIC 
# MAGIC Positve: The flight is delayed by 15 minutes. 
# MAGIC 
# MAGIC Negative: The flight is not delayed by 15 minutes
# MAGIC 
# MAGIC Pos / Neg will be determined by the value of the DEP_DEL15 Column. If the value is 1, the flight is delayed, if the value is 0, the flight is not delayed, if the value is null, the flight was cancelled.
# MAGIC 
# MAGIC True Positive: The flight was delayed and the model predicted it was delayed. 
# MAGIC 
# MAGIC False Positive: The flight was not delayed and the model predicted it was delayed. 
# MAGIC 
# MAGIC True Negative: The flight was not delayed and the model predicted it was not delayed
# MAGIC 
# MAGIC False Negative: The flight was delayed and the model predicted it was not delayed. 

# COMMAND ----------

# MAGIC %md
# MAGIC __Customer Experience__
# MAGIC 
# MAGIC True Positive: This is a good experience. The customer was correctly notified that the flight was delayed. 
# MAGIC 
# MAGIC False Positive: This is a bad experience. The customer expected a delay, but the flight was on time. this could mean missed flights or extra effort spent planning for delay
# MAGIC 
# MAGIC True Negative: This is a neutral experience. The customer was not notified of any delay and there was no delay. 
# MAGIC 
# MAGIC False Negative: This is a bad experience. The customer was not notified of any delay but there was a delay. This experience can also be considered a status quo, since this would be the experience if there was no model / prediction made. 

# COMMAND ----------

# MAGIC %md
# MAGIC __Possible Evaluation metrics__
# MAGIC 
# MAGIC Accuracy: Number of predictions correctly made | $$\frac{TP+TN}{TP+TN+FP+FN}$$
# MAGIC 
# MAGIC Pros: Easily understandable metric that shows how many predictions were correct from the total predictions. 
# MAGIC 
# MAGIC Cons: Easy to get a high value due to ~80% of flights being not delayed. 
# MAGIC 
# MAGIC ___
# MAGIC 
# MAGIC Precision: Number of positives correctly predicted out of all positive predictions | $$\frac{TP}{TP+FP}$$ 
# MAGIC 
# MAGIC Pros: Focuses on reducing false positives
# MAGIC 
# MAGIC Cons: Does not consider false negatives
# MAGIC 
# MAGIC ___
# MAGIC 
# MAGIC Recall: Number of positives correctly predicted out of all actual positives | $$\frac{TP}{TP+FN}$$
# MAGIC 
# MAGIC Pros: Focuses on reducing False negative. 
# MAGIC 
# MAGIC Cons: Less focus on reducing false positives
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC F1-Score: Harmonic mean between recall and precision | $$2 * \frac{Precision * Recall}{Precision+Recall}$$
# MAGIC 
# MAGIC Pros: Balances the benefits of Precision and Recall 
# MAGIC 
# MAGIC Cons: Difficult to calculate / Gives equal importance to precision and recall (may want to tweak the weights)
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC F-Score with B=0.5 : Weighted mean between recall and precision, with extra weight given to precision.  | $$ (1 + (0.5)^2) * \frac{Precision * Recall}{((0.5)^2 * Precision)+Recall}$$
# MAGIC 
# MAGIC Pros: Considers both False Negative and False Positive as improvement areas, but gives emphasis to False Positive as that is a worse customer experience. 
# MAGIC 
# MAGIC Cons: Difficult to calculate (compared to others)

# COMMAND ----------

# MAGIC %md
# MAGIC __Recommendation__
# MAGIC 
# MAGIC 1. Custom F-Score
# MAGIC 1. F1-Score
# MAGIC 1. Precision
# MAGIC 1. Recall
# MAGIC 1. Accuracy