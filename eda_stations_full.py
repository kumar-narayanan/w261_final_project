# Databricks notebook source
# MAGIC %md
# MAGIC ## EDA for Station Data

# COMMAND ----------

from pyspark.sql.functions import col,split,isnan, when, count
from pyspark.sql.functions import hour, minute, floor, to_date, lit
from pyspark.sql.types import IntegerType
import pandas as pd

blob_container = "team08-container" # The name of your container created in https://portal.azure.com
storage_account = "team08" # The name of your Storage account created in https://portal.azure.com
secret_scope = "team08-scope" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "team08-key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# COMMAND ----------

# Inspect the Mount's Final Project folder 
display(dbutils.fs.ls("/mnt/mids-w261/datasets_final_project/"))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Load Station Data

# COMMAND ----------

df_stations = spark.read.parquet("/mnt/mids-w261/datasets_final_project/stations_data/*")

# COMMAND ----------

df_stations.printSchema()

# COMMAND ----------

display(df_stations)

# COMMAND ----------

display(df_stations.summary())

# COMMAND ----------

df_stations.count()

# COMMAND ----------

# MAGIC %md ### Neighboring States

# COMMAND ----------

display(df_stations.select("neighbor_state").distinct())

# COMMAND ----------

# MAGIC %md ### Stations

# COMMAND ----------

df_stations.select("station_id").distinct().count()

# COMMAND ----------

df_stations.select("wban").distinct().count()

# COMMAND ----------

df_stations.select("usaf").distinct().count()

# COMMAND ----------

df1 = df_stations.select("neighbor_state").groupBy("neighbor_state").count()

display(df1.sort(col("count")))

# COMMAND ----------

display(df_stations
        .select("neighbor_state")
        .groupBy("neighbor_state")
        .count())


# COMMAND ----------

display(df_stations.where(col('neighbor_id') == col('station_id')))

# COMMAND ----------

# MAGIC %md
# MAGIC ![station_map](https://team08.blob.core.windows.net/team08-container/weather_station_map.png)