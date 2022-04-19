# Databricks notebook source
from pyspark.sql.functions import col, max

blob_container = "team08-container" # The name of your container created in https://portal.azure.com
storage_account = "team08" # The name of your Storage account created in https://portal.azure.com
secret_scope = "team08-scope" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "team08-key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# COMMAND ----------

df_airlines_3m = spark.read.parquet(f"{blob_url}/airline_data_clean2_3m")
df_airlines_6m = spark.read.parquet(f"{blob_url}/airline_data_clean2_6m")

# COMMAND ----------

def train_test_split(dataframe, train_weight, test_weight, seed = 2018):
    dates = dataframe.select(dataframe.FL_DATE_UTC).distinct().orderBy(dataframe.FL_DATE_UTC)
    
    train_dates, test_dates = dates.randomSplit([train_weight, test_weight], seed)
    train_dates = set(train_dates.toPandas()['FL_DATE_UTC'])
    test_dates = set(test_dates.toPandas()['FL_DATE_UTC'])
    
    train_df = dataframe.where(dataframe.FL_DATE_UTC.isin(train_dates))
    test_df = dataframe.where(dataframe.FL_DATE_UTC.isin(test_dates))
    
    return train_df, test_df

# COMMAND ----------

train_df, test_df = train_test_split(df_airlines_3m, 0.8, 0.2)

display(train_df)
display(test_df)

print(train_df.count())
print(test_df.count())
print(df_airlines_3m.count())

# COMMAND ----------

