import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import time
import datetime
import numpy as np
from datetime import date
from sklearn import preprocessing

import warnings
warnings.filterwarnings("ignore")



st.image("https://raw.githubusercontent.com/shubhadipbiswas/Classification_heroku/master/image.png",width=600)
st.write("""
        
This app predicts the **Booking Status!
""")
st.sidebar.header('Input Features Upload')



st.sidebar.markdown("""
[Example CSV input file](https://github.com/shubhadipbiswas/Classification_heroku/blob/master/sample.csv)
""")
# Collects user input features into dataframe
input_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if input_file is not None:
    input_df = pd.read_csv(input_file)
else:
    st.write('Awaiting CSV file to be uploaded.')
try:
    #input_df = pd.read_csv(input_file)
    raw = pd.read_csv('https://raw.githubusercontent.com/shubhadipbiswas/bookingapp_heroku/master/Booking_Classification.csv')
    input_1 = raw.drop(columns=['Booking_Cancelled'], axis=1)
    target = ['Booking_Cancelled']
    zeroencode = ['Hotel_Type','Boarding','Booking_Mode','Booking_Channel','Cat_Type1','Cat_Type2','deposit_type','customer_type']
    drops=['country', 'Day_Of_Month','Date_Month','date', 'agent', 'company', 'reservation_status', 'reservation_status_date']
    df = pd.concat([input_df,input_1],axis=0)
    df['month'] = pd.to_datetime(df.Date_Month, format='%B').dt.month
    cols=["Date_Year","month","Day_Of_Month"]
    df['date'] = df[cols].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")
    df['date']=pd.to_datetime(df['date'])
    df['weekday'] = df['date'].dt.dayofweek
    df['weekday'] = df['date'].apply(lambda x: x.weekday())
    df['Weekend'] = df.weekday.isin([5, 6]).astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df=df.drop(drops, axis = 1) 
    df= pd.get_dummies(df, columns=zeroencode)
    min_max_scaler = preprocessing.MinMaxScaler()
    df['lead_time']= min_max_scaler.fit_transform(df[['lead_time']])
    df['adr']= min_max_scaler.fit_transform(df[['adr']])
    df=df.replace([np.inf, -np.inf], np.nan)
    df=df.fillna(0)
    Length=len(input_df)
    df = df[:Length] # Selects only the rows of user input data
    # Displays the user input features
    st.subheader('User Input features')
    if input_file is not None:
        st.write(df)
    else:
        st.write('Awaiting CSV file to be uploaded.')
    
    # Reads in saved classification model
    load_clf = pickle.load(open('clas_rf.pkl', 'rb'))
    
    # Apply model to make predictions
    prediction = load_clf.predict(df)
    prediction_proba = load_clf.predict_proba(df)
    
    st.subheader('Booking Prediction')
    booing_status= np.array(['cancelled','booked'])
    st.write(booing_status[prediction])

    st.subheader('Probability of Predictions')
    st.write(prediction_proba)
except:
    st.write("""
             ### What are you waiting For?
             Please Upload!
             """)
                                         