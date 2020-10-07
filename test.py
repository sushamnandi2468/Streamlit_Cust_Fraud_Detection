import pandas as pd
import joblib
import os


path_to_artifacts = os.path.normpath(os.getcwd())
model = joblib.load(path_to_artifacts + "\FDTree.joblib")
wi_fn = joblib.load(path_to_artifacts + "\wi_fn.joblib")
wi_ln = joblib.load(path_to_artifacts + "\wi_ln.joblib")

fname_model_list = list(wi_fn.keys())
lname_model_list = list(wi_ln.keys())

df=pd.read_csv("test.csv", delimiter=",")

#print(fname_model_list)

df["First_Name"] = df["First_Name"].str.lower()
df["Last_Name"] = df["Last_Name"].str.lower()

df['Dedup'] = df.First_Name.isin(fname_model_list).astype(int)
df['Dedup'] = df.Last_Name.isin(lname_model_list).astype(int)
df1=df.iloc[:, :-1]
#print(df1.head(10))
#print(df.head(10))
df1 = df1.replace({"First_Name" : wi_fn})
df1 = df1.replace({"Last_Name" : wi_ln})

df1[['DD','MM','YYYY']]=df1.DOB.str.split("-", expand=True,)
#Now DOB column can be dropped from the dataframe
df1=df1.drop(columns=['DOB','Customer_Type', 'PAN', 'Deceased_Flag', 'Gender', 'Martial_Status', 'PEP_Flag', 'CTF_Flag', 'Country_of_residence', 'Country_of_Origin'])
df1['DD']=df1['DD'].astype(int)
df1['MM']=df1['MM'].astype(int)
df1['YYYY']=df1['YYYY'].astype(int)
#print(df1.head(10))
prediction = model.predict_proba(df1.iloc[4:5, :])[0]
print(prediction.type())
