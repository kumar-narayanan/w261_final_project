# Databricks notebook source
# MAGIC %md # Lookup Airport Geolocation

# COMMAND ----------

import geopy
from geopy.geocoders import Nominatim
import numpy as np
import pandas as pd

# COMMAND ----------

# MAGIC %md ## Read US airport codes

# COMMAND ----------

df = pd.read_csv('airport_code_usa.csv')
df.head()

# COMMAND ----------

df['Airport_code'] = df['Code'] + ' airport'

# COMMAND ----------

df.head()

# COMMAND ----------

df.shape

# COMMAND ----------

def get_lat_lng(address, geolocator):
    location = geolocator.geocode(address)
    #print(location.address)
    if location is None:
        return (0.0,0.0)
    else:
        return (location.latitude, location.longitude) 

# COMMAND ----------

geoloc = Nominatim(user_agent="kumarn@berkeley.edu")
address = 'SFO airport'
lat_lng = get_lat_lng(address, geoloc)
print(lat_lng)

# COMMAND ----------

df['Lat_Lng'] = df.apply(lambda row: get_lat_lng(row.Airport_code, geoloc), axis = 1)

# COMMAND ----------

is_NaN = df.isnull()
row_has_NaN = is_NaN.any(axis=1)
df[row_has_NaN]

# COMMAND ----------

df.head()

# COMMAND ----------

df1.head()

# COMMAND ----------

lat_lng_list = list(df['Lat_Lng'])

# COMMAND ----------

lat = [x[0] for x in lat_lng_list]
lng = [x[1] for x in lat_lng_list]

# COMMAND ----------

df['Lat'] = lat
df['Lng'] = lng

# COMMAND ----------

df.head()

# COMMAND ----------

# MAGIC %md ## Save Airport Geolocation lookup data

# COMMAND ----------

df.to_csv('airport_code_lat_lng.csv')

# COMMAND ----------

df_sta = pd.read_csv('station.csv')
df_sta.head()

# COMMAND ----------

loc = df_sta[['lat', 'lon']]
fig = gmaps.figure()
fig.add_layer(gmaps.heatmap_layer(loc))
fig