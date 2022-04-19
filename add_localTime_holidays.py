# Databricks notebook source
# DBTITLE 1,All imports
import airporttime
import datetime as dt
from datetime import datetime
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType, TimestampType, StringType
import math
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

import pyspark.sql.functions as F
from pyspark.sql.functions import col, split, isnan, when, count, concat, lit, desc, floor, max, avg, min, regexp_replace, first
from pyspark.sql.functions import hour, minute, floor, to_date, rand
from pyspark.sql.functions import array, create_map, struct
from pyspark.sql.functions import substring, lit, udf, row_number, lower
from pyspark.sql.functions import sum as ps_sum, count as ps_count
from pyspark.sql import SQLContext
from pyspark.sql import DataFrameNaFunctions
from pyspark.sql import DataFrame
from pyspark.sql import Row

# COMMAND ----------

# DBTITLE 1,Environment
blob_container = "team08-container" # The name of your container created in https://portal.azure.com
storage_account = "team08" # The name of your Storage account created in https://portal.azure.com
secret_scope = "team08-scope" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "team08-key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# COMMAND ----------

# DBTITLE 1,Read the newly created data
df_airlines_weather_new = spark.read.parquet(f"{blob_url}/df_airlines_weather_updated")

# COMMAND ----------

display(df_airlines_weather_new)

# COMMAND ----------

# DBTITLE 1,Function to convert UTC to local time zone
def convert_airport_to_local(airport, fl_time_utc, ret_unit='date'): 
    apt = airporttime.AirportTime(iata_code=airport)
    #utc_time = datetime(int(fl_time_utc[0:4]), int(fl_time_utc[5:7]), int(fl_time_utc[8:10]), int(fl_time_utc[11:13]), int(fl_time_utc[14:16]))
    local_time = apt.from_utc(fl_time_utc)
    if ret_unit == 'date':
        return str(local_time)
    if ret_unit == 'hour':
        return str(local_time.hour)
    if ret_unit == 'year':
        return str(local_time.year)
    if ret_unit == 'date_only':
        return str(local_time)[0:10]

convert_airport_to_local_udf = udf(convert_airport_to_local, StringType())

# COMMAND ----------

# DBTITLE 1,USA holiday calendar
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start='2015-01-01', end='2020-12-31')
holidays_num_dict = {}
for i in range(len(holidays)):
    key = datetime.strftime(holidays[i], '%Y-%m-%d')
    holidays_num_dict[key] = 1 # the day is a holiday
    
    if key[5:10] == '01-01': # new year 
        prev = 1
        next = -1
    elif key[5:9] == '11-2': # thanksgiving holiday
        prev = 1
        next = -3
    elif holidays[i].weekday() == 0: # mon. is a holiday
        prev = 3
        next = -1
    else: #any other day for the holiday
        prev = 1
        next = -1
        
    key = datetime.strftime(holidays[i] - dt.timedelta(days=prev), '%Y-%m-%d')
    holidays_num_dict[key] = 2 # the date is a logical previous day
    
    key = datetime.strftime(holidays[i] - dt.timedelta(days=next), '%Y-%m-%d')
    holidays_num_dict[key] = 3 # the date is a logical next day
    
def getHolidayNum(date_str):
    return holidays_num_dict.get(date_str, 0)

getHolidayNum_udf = udf(getHolidayNum, IntegerType())

# COMMAND ----------

# DBTITLE 1,All columns Year, Date_only, FL_DATE_LOCAL (Flight dep. in local time)
df_airlines_weather_new = df_airlines_weather_new.withColumn("FL_DATE_LOCAL", convert_airport_to_local_udf(df_airlines_weather_new.ORIGIN, df_airlines_weather_new.FL_DATETIME_UTC)) \
                                                 .withColumn("Year", convert_airport_to_local_udf(df_airlines_weather_new.ORIGIN, df_airlines_weather_new.FL_DATETIME_UTC, lit('year'))) \
                                                 .withColumn("Date_only", convert_airport_to_local_udf(df_airlines_weather_new.ORIGIN, df_airlines_weather_new.FL_DATETIME_UTC, lit('date_only')))

# COMMAND ----------

# DBTITLE 1,Add a col to indicate holiday (1-holiday, 2-pre-holiday busy travel day, 3-post-holiday busy travel day, 0-normal day)
df_airlines_weather_new = df_airlines_weather_new.withColumn("isHoliday", getHolidayNum_udf(df_airlines_weather_new.Date_only))

# COMMAND ----------

display(df_airlines_weather_new)

# COMMAND ----------

# DBTITLE 1,Save to a parquet file
df_airlines_weather_new.write.mode('overwrite').parquet(f"{blob_url}/df_airlines_weather_holidays")