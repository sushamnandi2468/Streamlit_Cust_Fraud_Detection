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

st.sidebar.title("Filter data")
st.set_option('deprecation.showfileUploaderEncoding', False)
#Get User Inputs
def get_user_input():
    
    uploaded_file = st.file_uploader("Choose a CSV file for processing", type='csv')
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        st.stop()        
    return data

user_input = get_user_input()

#Display User Input
#st.subheader('User Input:')
#st.write(user_input)

#Load machine learning models
path_to_artifacts = os.path.normpath(os.getcwd())
model = joblib.load(path_to_artifacts + "\FDTree_CFD.joblib")
df_country_dict = joblib.load(path_to_artifacts + "\df_country_dict.joblib")
wi_fn = joblib.load(path_to_artifacts + "\wi_fn_CFD.joblib")
wi_ln = joblib.load(path_to_artifacts + "\wi_ln_CFD.joblib")

#Data Pre processing
def preprocessing(input_data):
    # JSON to pandas DataFrame
    #input_data = pd.DataFrame(input_data, index=[0])
    input_data["First_Name"] = input_data["First_Name"].str.lower()
    input_data["Last_Name"] = input_data["Last_Name"].str.lower()

    try:       
        input_data = input_data.replace({"First_Name" : wi_fn})
        input_data = input_data.replace({"Last_Name" : wi_ln})
        #st.write("Names replaced")
    except Exception:
        return {"status": "Error", "message": "Error in conversion"}
        
    #DOB to be split into DD MM and YYYY for ML algo
    input_data[['DD','MM','YYYY']]=input_data.DOB.str.split("-", expand=True,)
    #Now DOB column can be dropped from the dataframe
    input_data=input_data.drop(columns='DOB')
    input_data['DD']=input_data['DD'].astype(int)
    input_data['MM']=input_data['MM'].astype(int)
    input_data['YYYY']=input_data['YYYY'].astype(int)
    #st.write("DOB expanded")

    #input_data = input_data.replace({"Customer_Type" : cust_type})
    #st.write("Cust_Type replaced")
    deceased_flag={
        True : 1,
        False : 0
    }

    input_data = input_data.replace({'Deceased_flag' : deceased_flag})

    gender ={
        'M' : 0,
        'F' : 1
    }

    input_data = input_data.replace({"Gender" : gender})

    martial_status={
        'Married' : 1,
        'Not Married' : 0
    }

    input_data = input_data.replace({"Martial_Status" : martial_status})

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

    input_data = input_data.replace({'Country_of_residence' : df_country_dict})
    input_data = input_data.replace({'Country_of_Origin' : df_country_dict})
    st.write(input_data)
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
    try:
        actual_data = input_data
        input_data = preprocessing(input_data)
        #st.write("Input data preprocessed")
        pred_full={}
        for i in input_data.index:
            #st.write(predict(input_data[i:i+1]))
            prediction = predict(input_data[i:i+1])[0]  # for the complete file
            #st.write("Prediction is", prediction)
            prediction = postprocessing(prediction)
            #st.write("Prediction post processing is", prediction)
            pred_full.update({ i : prediction['label']})
        #st.write("Final Dict", pred_full)    
        df_pred=pd.DataFrame(list(pred_full.items()), columns=['id','label'])            
        df_pred.drop(columns='id')
        #st.write("Latest Prediction post processing is", df_pred)        
        output_data = pd.concat([input_data, df_pred.reindex(input_data.index)], axis=1)
        output_data['First_Name']=actual_data['First_Name']
        output_data['Last_Name']=actual_data['Last_Name']

    except Exception as e:
        return {"status": "Error", "message": str(e)}

    return output_data

#Predcition
prediction=compute_prediction(user_input)
st.subheader('Classification:')

# function for set text color of False Positive 
# values in Dataframes 
def color_positive_red(val): 
	""" 
	Takes a scalar and returns a string with 
	the css property `'color: green'` for positive 
	strings, black otherwise. 
	"""
	if val == 'False Positive': 
		color = 'green'
	else: 
		color = 'black'
	return 'color: %s' % color 

st.write(prediction[['First_Name', 'Last_Name', 'PAN' ,'label']])
plt.title('Number of Flase Positives Captured')
plt.xlabel('Classification')
plt.ylabel('Count')

# fix the legend
current_handles, _ = plt.gca().get_legend_handles_labels()
reversed_handles = reversed(current_handles)

labels = reversed(prediction['label'].unique())

plt.legend(reversed_handles,labels,loc='lower right')

prediction.groupby('label')['PAN'].nunique().plot(kind='bar')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()