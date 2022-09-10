import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
# import plotly.express as px
import pickle
from pickle import load
import os
import base64


pwd = os.getcwd()

st.set_page_config(
    page_title="Rossman Pharmaceuticals Sales Prediction", layout="wide")

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
        unsafe_allow_html=True
    )


add_bg_from_local('images/19366.jpg')

def data_prep(input):
    cols = ['Date','StateHoliday','StoreType','Assortment','Store','Promo','SchoolHoliday',
                                           'Promo2','Season']
    
    row = pd.DataFrame(data=[input],columns=cols)
    row['Date'] = pd.to_datetime(row['Date'])
    row['Year'] = row['Date'].dt.year
    row['Month'] = row['Date'].dt.month
    row['Quarter'] = row['Date'].dt.quarter
    row['Week'] = row['Date'].dt.week
    row['Day'] = row['Date'].dt.day
    row['WeekOfYear'] = row['Date'].dt.weekofyear
    row['DayOfYear'] = row['Date'].dt.dayofyear
    row['DayOfWeek'] = row['Date'].dt.day_of_week
    row['IsWeekDay'] = [int(row.Date.dt.day_of_week < 5)]
    row['CompetitionOpenMonthDuration'] = [np.random.randint(0,12)]
    row['PromoOpenMonthDuration'] = [np.random.randint(0,12)]
    row['Month_Status'] = ['Beginning']
    row['CompetitionDistance'] = [np.random.randint(0,120)]
    row['PromoInterval'] = ['0,0,0,0']
    row['Until_Holiday'] = [np.random.randint(0,12)]
    row['Since_Holiday'] = [np.random.randint(0,12)]
    row['Open'] = [1]
    
    prep = load(open('models/data_transformer.pkl', 'rb'))
    row.drop(columns=['Date'],inplace=True,axis=1)
    return prep.transform(row)
    
    
def compute_result(input):
    inp = data_prep(input)
    filename = "models/10-09-2022_15_42_47_.pkl"
    model = pickle.load(open(filename, 'rb'))
    return model.predict(inp)
    




st.title("Rossman Pharmaceuticals Sales Prediction")
col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    with st.form("my_form"):
        st.write("Insert the Features to Predict Sales")
        
        date = st.date_input("Date Input")
        stateholiday = st.selectbox("State Holiday", ["0","a","b","c"])
        season = st.selectbox("Season", ["Spring","Winter","Summer","Autumn"])
        sachoolholiday = st.checkbox("School Holiday")
        storetype = st.selectbox("Store Type",["a","b","c","d"])
        assortment = st.selectbox("Assortment",["a","b","c"])
        compdis = st.number_input("Competition Distance",0,1000)
        store = st.number_input("Store Number", 1, 1115)
        promo = st.checkbox("Promotion Available")
        promo2 = st.checkbox("Promotion Available Today")

        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted:
            input=[date,stateholiday,storetype,assortment,store,int(promo),int(sachoolholiday),int(promo2),season]
            result = compute_result(input)
            st.write("Prediction result:", result[0])

with col3:
    st.write(' ')



st.write("Outside the form")


