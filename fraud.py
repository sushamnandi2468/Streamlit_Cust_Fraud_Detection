#Customer Fraud Detection WebApp

import pandas as pd
import joblib
import os
import streamlit as st
import datetime

#Title
st.write("""
# Customer Fraud Detection
""")

st.sidebar.title("Filter data")

#Get User Inputs
def get_user_input():
    first_name = st.sidebar.text_input("First Name", "Osama")
    last_name = st.sidebar.text_input("Last Name", "ElLaden")
    dob = st.sidebar.date_input("Date of Birth")
    dob= dob.strftime("%d-%m-%Y")
    
    customer_type=['Individual', 'Retail Customer', 'Ship','Private Organization','Private Organization']
    cust_type = st.sidebar.selectbox("Customer Type",customer_type, index=0)

    pan=st.sidebar.text_input("PAN","123456789")
    deceased=st.sidebar.selectbox("Deceased",('Yes', 'No'))
    gender=st.sidebar.selectbox("Gender",('Male', 'Female'))
    martial_status=st.sidebar.selectbox("Martial Status",('Married', 'Not Married'))
    pep=st.sidebar.selectbox("PEP Flag",('Yes', 'No'))
    ctf=st.sidebar.selectbox("CTF Flag",('Yes', 'No'))
    country=pd.read_csv('Country.csv', delimiter=',')
    cor=st.sidebar.selectbox("Country Of Residence",country)
    coo=st.sidebar.selectbox("Country Of Origin",country)

    #Store data in a dictionary
    user_data = {
        'First_Name': first_name,
        'Last_Name': last_name,
        'DOB':dob,
        'Customer_Type': cust_type,
        'pan' : pan,
        'Deceased_flag' : deceased,
        'gender' : gender,
        'martial_status' : martial_status,
        'pep' : pep,
        'ctf' : ctf,
        'cor' : cor,
        'coo' : coo
    }
    
    #Transform data into dataframe
    features = pd.DataFrame(user_data, index=[0])
    return features

user_input = get_user_input()

#Display User Input
st.subheader('User Input:')
st.write(user_input)

#Load machine learning models
path_to_artifacts = os.path.normpath(os.getcwd())
model = joblib.load(path_to_artifacts + "\FDTree_CFD.joblib")
df_country_dict = joblib.load(path_to_artifacts + "\df_country_dict.joblib")
wi_fn = joblib.load(path_to_artifacts + "\wi_fn_CFD.joblib")
wi_ln = joblib.load(path_to_artifacts + "\wi_ln_CFD.joblib")

#Data Pre processing
def preprocessing(input_data):
    # JSON to pandas DataFrame
    input_data = pd.DataFrame(input_data, index=[0])
           
    input_data = input_data.replace({"First_Name" : wi_fn})
    input_data = input_data.replace({"Last_Name" : wi_ln})
        
    #DOB to be split into DD MM and YYYY for ML algo
    input_data[['DD','MM','YYYY']]=input_data.DOB.str.split("-", expand=True,)
    #Now DOB column can be dropped from the dataframe
    input_data=input_data.drop(columns='DOB')
    input_data['DD']=input_data['DD'].astype(int)
    input_data['MM']=input_data['MM'].astype(int)
    input_data['YYYY']=input_data['YYYY'].astype(int)

    cust_type={
        'Individual': 1,
        'Retail Customer' : 2,
        'Ship' : 3,
        'Private Organization' : 4,
        'Private Organization' : 5
    }

    input_data = input_data.replace({"Customer_Type" : cust_type})

    deceased_flag={
        'Yes' : 1,
        'No' : 0
    }

    input_data = input_data.replace({'Deceased_flag' : deceased_flag})

    gender ={
        'Male' : 0,
        'Female' : 1
    }

    input_data = input_data.replace({"gender" : gender})

    martial_status={
        'Married' : 1,
        'Not Married' : 0
    }

    input_data = input_data.replace({"martial_status" : martial_status})

    pep_flag={
        'No' : 0,
        'Yes' : 1
    }

    input_data = input_data.replace({'pep': pep_flag})

    ctf_flag={
        'No' : 0,
        'Yes' : 1
    }

    input_data = input_data.replace({'ctf': ctf_flag})

    input_data = input_data.replace({'cor' : df_country_dict})
    input_data = input_data.replace({'coo' : df_country_dict})

    return input_data

#user_input_pr=preprocessing(user_input)
def predict(input_data):
    return model.predict_proba(input_data)
        
def postprocessing(input_data):
    if input_data[1] == 1:
        label = 'Fraud'
    else :
        label = 'Not Fraud'
    return {"probability": input_data[1], "label": label, "status": "OK"}
        
def compute_prediction(input_data):
    try:
        input_data = preprocessing(input_data)
        st.write(input_data)
        prediction = predict(input_data)[0]  # only one sample
        prediction = postprocessing(prediction)
    except Exception as e:
        return {"status": "Error", "message": str(e)}

    return prediction

#Predcition
prediction=compute_prediction(user_input)
st.subheader('Classification:')
st.write(prediction["label"])