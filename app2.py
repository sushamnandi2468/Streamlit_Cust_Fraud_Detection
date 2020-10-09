#Customer Fraud Detection WebApp File Upload

import pandas as pd
import joblib
import os
import streamlit as st
import datetime
from nltk.tokenize.treebank import TreebankWordDetokenizer as wd
import seaborn as sns
import base64
from pathlib import Path
import matplotlib.pyplot as plt

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

header_html = "<img src='data:image/png;base64,{}' class='img-fluid' width='300' height='240'>".format(
    img_to_bytes("CG_Invent.PNG")
)
st.markdown(
    header_html, unsafe_allow_html=True,
)

sidebar_html = "<img src='data:image/png;base64,{}' class='img-fluid' width='200' height='160'>".format(
    img_to_bytes("DRR.PNG")
)
st.sidebar.markdown(
    sidebar_html, unsafe_allow_html=True,
)

#Title
st.write("""
# Customer Fraud Detection
""")

#Lookup Single User Data

#Load machine learning models
path_to_artifacts = os.path.normpath(os.getcwd())
model = joblib.load(path_to_artifacts + "\FDTree.joblib")
wi_fn = joblib.load(path_to_artifacts + "\wi_fn.joblib")
wi_ln = joblib.load(path_to_artifacts + "\wi_ln.joblib")

def get_lookup_input():
    first_name = st.text_input("First Name", "SAED")
    last_name = st.text_input("Last Name", "ELHOORIE")
    dob = st.date_input("Date of Birth")
    dob= dob.strftime("%d-%m-%Y")
    #Store data in a dictionary
    user_data = {
        'First_Name': first_name,
        'Last_Name': last_name,
        'DOB':dob
    }
    
    #Transform data into dataframe
    features = pd.DataFrame(user_data, index=[0])
    return features

lookup_input = get_lookup_input()

#Data Pre processing
def lookup_preprocessing(input_data):
    # JSON to pandas DataFrame
    input_data = pd.DataFrame(input_data, index=[0])
    input_data["First_Name"] = input_data["First_Name"].str.lower()
    input_data["Last_Name"] = input_data["Last_Name"].str.lower()       
    input_data = input_data.replace({"First_Name" : wi_fn})
    input_data = input_data.replace({"Last_Name" : wi_ln})
        
    #DOB to be split into DD MM and YYYY for ML algo
    input_data[['DD','MM','YYYY']]=input_data.DOB.str.split("-", expand=True,)
    #Now DOB column can be dropped from the dataframe
    input_data=input_data.drop(columns='DOB')
    input_data['DD']=input_data['DD'].astype(int)
    input_data['MM']=input_data['MM'].astype(int)
    input_data['YYYY']=input_data['YYYY'].astype(int)
        
    return input_data

#user_input_pr=preprocessing(user_input)
def lookup_predict(input_data):
    return model.predict_proba(input_data)
        
def lookup_postprocessing(input_data):
    if input_data[1] == 1:
        label = 'Fraud'
    else :
        label = 'Not Fraud'
    return {"probability": input_data[1], "label": label, "status": "OK"}
        
def lookup_compute_prediction(input_data):
    try:
        input_data = lookup_preprocessing(input_data)
        #st.write(input_data)
        prediction = lookup_predict(input_data)[0]  # only one sample
        prediction = lookup_postprocessing(prediction)
    except Exception as e:
        return {"status": "Error", "message": str(e)}

    return prediction

#Predcition
lookup_prediction=lookup_compute_prediction(lookup_input)
st.subheader('Classification:')
st.write(lookup_prediction['label'])