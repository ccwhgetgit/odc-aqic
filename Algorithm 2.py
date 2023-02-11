#!/usr/bin/env python
# coding: utf-8

# Build and publish an algorithm to predict the concentration of one pollutant of your choice for each hour of the day from February 15 to 28 - on average for all stations. (20 points)

# In[1]:


pip install fbProphet


# In[2]:


import pandas as pd 
import numpy as np 

data = pd.read_csv('data.csv')
data['DATA'] = pd.to_datetime(data['DATA'])
data.head()


# In[3]:


data['month'] = data['DATA'].dt.month
data['year'] = data['DATA'].dt.year
hours = [ '01h', '02h', '03h', '04h', '05h', '06h', '07h', '08h',
       '09h', '10h', '11h', '12h', '13h', '14h', '15h', '16h', '17h', '18h',
       '19h', '20h', '21h', '22h', '23h', '24h']


# In[4]:


contaminant = data['CONTAMINANT'][0]
condition = data['CONTAMINANT'] == contaminant 
hours = [ '01h', '02h', '03h', '04h', '05h', '06h', '07h', '08h',
       '09h', '10h', '11h', '12h', '13h', '14h', '15h', '16h', '17h', '18h',
       '19h', '20h', '21h', '22h', '23h', '24h']
pollutant_data = data[hours]
pollutant_data.head()


# In[5]:


data = pd.read_csv('data.csv')
data.head()
data['DATA'] = pd.to_datetime(data['DATA'])
vals = pollutant_data.stack().values
new_df = pd.DataFrame(columns=['ds', 'y'])
new_df['y'] = vals 
new_df['ds'] = data['DATA']
new_df = new_df.iloc[::-1].tail(500000).reset_index(drop=True)
new_df


# In[6]:


from prophet import Prophet
model = Prophet(weekly_seasonality=True, daily_seasonality=True )
model.fit(new_df)


# In[7]:


from datetime import datetime, timedelta
  
# Create Custom Function
def date_range(start, end):
    delta = end - start
    days = [start + timedelta(days=i) for i in range(delta.days + 1)]
    return days
  
startDate = datetime(2023, 1, 26)
endDate = datetime(2023, 2, 28)
      
datesRange = date_range(startDate, endDate);
dates = pd.to_datetime(datesRange)
all_dates = []
for date in datesRange:
    for hour in range(24):
        date1 = str(date).split(" ")[0].split("-")
        yy = int(date1[0])
        mm = int(date1[1])
        dd = int(date1[2])
        time = datetime(yy, mm, dd, hour, 0, 0 )
        all_dates.append(time)
future = pd.DataFrame()
future['ds'] = all_dates
future


# In[8]:


predicted_vals = model.predict(future)
predicted_vals[['ds', 'yhat']]
start_date = datetime(2023, 2, 15, 0,0,0)
condition = predicted_vals['ds'] >= start_date
output = predicted_vals.loc[condition].reset_index(drop=True)
output[['ds','yhat']]


# In[9]:


output[['ds','yhat']].to_csv('algo_2.csv')

