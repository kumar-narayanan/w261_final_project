# Databricks notebook source
from datetime import datetime
test_data = [
    {"id": 1, "origin": "ORD", "DEST": "NYC", "carrier": "Delta", "tail_num": "A11", "flight_time": datetime(2022, 4, 2, 1, 30)},
    {"id": 2, "origin": "ORD", "DEST": "SFO", "carrier": "United", "tail_num": "A12", "flight_time": datetime(2022,4, 2, 2, 12)}, 
    {"id": 3, "origin": "ORD", "DEST": "LAX", "carrier": "Delta", "tail_num": "A13", "flight_time": datetime(2022, 4, 2, 3, )},
    {"id": 4, "origin": "ATL", "DEST": "PHI", "carrier": "United", "tail_num": "B11", "flight_time": datetime(2022, 4, 2, 1)},
    {"id": 5, "origin": "ATL", "DEST": "NYC", "carrier": "Delta", "tail_num": "C12", "flight_time": datetime(2022, 4, 2, 2)},
    {"id": 6, "origin": "ORD", "DEST": "SFO", "carrier": "United", "tail_num": "A11", "flight_time": datetime(2022,4, 2, 4)}, 
    {"id": 7, "origin": "ORD", "DEST": "LAX", "carrier": "Delta", "tail_num": "D23", "flight_time": datetime(2022, 4, 2, 5)},
    {"id": 8, "origin": "ATL", "DEST": "PHI", "carrier": "United", "tail_num": "S12", "flight_time": datetime(2022, 4, 2, 3)},
    {"id": 9, "origin": "ATL", "DEST": "NYC", "carrier": "Delta", "tail_num": "B11", "flight_time": datetime(2022, 4, 2, 4)},
    {"id": 10, "origin": "ORD", "DEST": "SFO", "carrier": "United", "tail_num": "A12", "flight_time": datetime(2022,4, 2, 6)}, 
    {"id": 11, "origin": "ORD", "DEST": "LAX", "carrier": "Delta", "tail_num": "E23", "flight_time": datetime(2022, 4, 2, 7)},
    {"id": 12, "origin": "ATL", "DEST": "PHI", "carrier": "United", "tail_num": "D23", "flight_time": datetime(2022, 4, 2, 5)},
    {"id": 13, "origin": "ORD", "DEST": "NYC", "carrier": "Delta", "tail_num": "A11", "flight_time": datetime(2022, 4, 2, 8)},
    {"id": 14, "origin": "ORD", "DEST": "SFO", "carrier": "United", "tail_num": "S34", "flight_time": datetime(2022,4, 2, 9)}, 
    {"id": 15, "origin": "ORD", "DEST": "LAX", "carrier": "Delta", "tail_num": "F21", "flight_time": datetime(2022, 4, 2, 10)},
    {"id": 16, "origin": "ATL", "DEST": "PHI", "carrier": "United", "tail_num": "A13", "flight_time": datetime(2022, 4, 2, 6)},
    {"id": 17, "origin": "ORD", "DEST": "NYC", "carrier": "Delta", "tail_num": "B11", "flight_time": datetime(2022, 4, 2, 11)},
    {"id": 18, "origin": "ORD", "DEST": "SFO", "carrier": "United", "tail_num": "C12", "flight_time": datetime(2022,4, 2, 12)}, 
    {"id": 19, "origin": "ORD", "DEST": "LAX", "carrier": "Delta", "tail_num": "A11", "flight_time": datetime(2022, 4, 2, 13)},
    {"id": 20, "origin": "ATL", "DEST": "PHI", "carrier": "United", "tail_num": "S12", "flight_time": datetime(2022, 4, 2, 6)},
]

# COMMAND ----------

test_df = spark.createDataFrame(test_data)

# COMMAND ----------

display(test_df)

# COMMAND ----------

test_df.printSchema()

# COMMAND ----------

from pyspark.sql.functions import desc

display(test_df.sort("origin", "flight_time"))

# COMMAND ----------

test_df.createOrReplaceTempView("test_view")

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number

display(spark.sql("""SELECT *, count(origin) OVER (
        PARTITION BY origin 
        ORDER BY flight_time 
        RANGE BETWEEN INTERVAL 2 HOURS PRECEDING AND CURRENT ROW
     ) AS number_of_flights FROM test_view"""))
