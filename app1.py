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

#Title
st.write("""
# Customer Fraud Detection
""")

#Get User Inputs
def get_user_input():
    
    uploaded_file = st.file_uploader("Choose a CSV file for processing", type='csv')
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        st.stop()        
    return data
st.set_option('deprecation.showfileUploaderEncoding', False)
user_input = get_user_input()

#Display User Input
st.subheader('User Input:')
if st.checkbox('Show User Input'):
    st.write(user_input)

#Load machine learning models
path_to_artifacts = os.path.normpath(os.getcwd())
model = joblib.load(path_to_artifacts + "\FDTree.joblib")
wi_fn = joblib.load(path_to_artifacts + "\wi_fn.joblib")
wi_ln = joblib.load(path_to_artifacts + "\wi_ln.joblib")

#Identify New Customers
def new_customer_identification(input_data):
    fname_model_list = list(wi_fn.keys())
    lname_model_list = list(wi_ln.keys())
    input_data["First_Name"] = input_data["First_Name"].str.lower()
    input_data["Last_Name"] = input_data["Last_Name"].str.lower()
    input_data['Dedup'] = input_data.First_Name.isin(fname_model_list).astype(int)
    input_data['Dedup'] = input_data.Last_Name.isin(lname_model_list).astype(int)

    return input_data   

#Data Pre processing
def preprocessing(input_data):
    #input_data["First_Name"] = input_data["First_Name"].str.lower()
    #input_data["Last_Name"] = input_data["Last_Name"].str.lower()
    #DOB to be split into DD MM and YYYY for ML algo
    input_data[['DD','MM','YYYY']]=input_data.DOB.str.split("-", expand=True,)
    input_data['DD']=input_data['DD'].astype(int)
    input_data['MM']=input_data['MM'].astype(int)
    input_data['YYYY']=input_data['YYYY'].astype(int)
    #Now DOB column can be dropped from the dataframe
    input_data=input_data.drop(columns=['DOB','Customer_Type', 'PAN', 'Deceased_Flag', 'Gender', 'Martial_Status', 'PEP_Flag', 'CTF_Flag', 'Country_of_residence', 'Country_of_Origin'])

    input_data = new_customer_identification(input_data)

    try:       
        input_data = input_data.replace({"First_Name" : wi_fn})
        input_data = input_data.replace({"Last_Name" : wi_ln})
        #st.write("Names replaced")
    except Exception:
        return {"status": "Error", "message": "Error in conversion"}
    
    #cols = list(input_data.columns)
    #cols = [cols[-1]] + cols[:-1]
    #input_data=input_data[cols]
    #st.write(input_data)

    return input_data

#user_input_pr=preprocessing(user_input)
def predict(input_data):
    return model.predict_proba(input_data)
        
def postprocessing(input_data):
    if input_data[1] == 0:
        label = 'False Positive'
    else :
        label = 'Fraud'
    return {"probability": input_data[1], "label": label, "status": "OK"}
        
def compute_prediction(input_data):
    actual_data = input_data
    input_data = preprocessing(input_data)
    pred_full={}
    for i in input_data.index:
        #st.write(predict(input_data[i:i+1]))
        if input_data.at[i,'Dedup'] == 1:
            prediction = predict(input_data.iloc[i:i+1, :-1])[0]  # for the complete file
            #st.write("Prediction is", prediction)
            prediction = postprocessing(prediction)
            #st.write("Prediction post processing is", prediction)
            pred_full.update({ i : prediction['label']})
        else :
            label = 'New Customer'
            prediction= {"probability": 2, "label": label, "status": "OK"}
            pred_full.update({ i : prediction['label']})

    #st.write("Final Dict", pred_full)    
    df_pred=pd.DataFrame(list(pred_full.items()), columns=['id','label'])            
    df_pred.drop(columns='id')
    #st.write("Latest Prediction post processing is", df_pred)        
    output_data = pd.concat([input_data, df_pred.reindex(input_data.index)], axis=1)
    output_data['First_Name']=actual_data['First_Name']
    output_data['Last_Name']=actual_data['Last_Name']
    #output_data = output_data.drop(columns='Dedup')

    return output_data

#Predcition
prediction=compute_prediction(user_input)
st.subheader('Classification:')
if st.checkbox('Show Classification'):
    st.table(prediction[['First_Name', 'Last_Name' ,'label']])

plt.title('Number of Flase Positives Captured')
plt.xlabel('Classification')
plt.ylabel('Count')

# fix the legend
current_handles, _ = plt.gca().get_legend_handles_labels()
reversed_handles = reversed(current_handles)

labels = reversed(prediction['label'].unique())

plt.legend(reversed_handles,labels,loc='lower right')

prediction.groupby('label')['First_Name'].nunique().plot(kind='bar')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()