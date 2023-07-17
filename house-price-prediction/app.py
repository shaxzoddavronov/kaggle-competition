import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from data_preprocessing import prepared
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from PIL import Image
from math import sqrt

img=Image.open('house.png')
file_path='model.pkl'
train=pd.read_csv('train.csv',index_col=0)
test=pd.read_csv('test.csv')
train1=train.copy()
test1=test.copy()
with open(file_path,'rb') as file:
    model=pickle.load(file)

train_prep,test_prep=prepared(train,test)
st.image(img)
st.title('House Price Prediction')
st.subheader('Description')
st.write("""Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement
          ceiling or the proximity to an east-west railroad.But this playground competition's dataset proves that much more 
          influences price negotiations than the number of bedrooms or a white-picket fence.With 79 explanatory variables
          describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict 
          the final price of each home.""")
st.subheader('Acknowledgments')
st.write("""The Ames Housing dataset was compiled by Dean De Cock for use in data science education. It's an incredible 
          alternative for data scientists looking for a modernized and expanded version of the often cited Boston Housing dataset. """)

st.markdown("""**How to evaluate?** 
* **Just enter row number**
* **Result will be shown with RMSE value**(Only for train set)""")


def rmse(observ,pred): return sqrt((observ-pred)**2)

def filtering(data):
    st.sidebar.subheader('Train dataset filtering')        
    yearBuilt=st.sidebar.slider('Year built',int(data['YearBuilt'].min()),int(data['YearBuilt'].max()))
    yrSold=st.sidebar.slider('Year sold',int(data['YrSold'].min()),int(data['YrSold'].max()))
    styles_unique=data['HouseStyle'].unique()
    styles=st.sidebar.multiselect('House styles',styles_unique,styles_unique)
    zoning_unique=data['MSZoning'].unique()
    zoning=st.sidebar.multiselect('General zoning',zoning_unique,zoning_unique)
    ngh_unique=data['Neighborhood'].unique()
    neighborhood=st.sidebar.multiselect('Neighborhood',ngh_unique,ngh_unique)
    data=data[(data['YearBuilt']<=yearBuilt)&(data['YrSold']<yrSold)&
              (data['HouseStyle'].isin(styles))&(data['MSZoning'].isin(zoning))&(data['Neighborhood'].isin(neighborhood))]
    data.index=np.arange(1,len(data)+1)
    return data

def evaluate(data,data_prep,sub):    
    st.subheader(sub)
    st.dataframe(data,height=500)
    row=st.number_input('Imput row number',0,len(data))
    prediction=model.predict(data_prep[row].reshape(1,-1))
    st.success(int(prediction))
    if data  is not test:
        st.info(f"RMSE: {rmse(data['SalePrice'][row+1],prediction):.2f}")
            
train=filtering(train)    
evaluate(train,train_prep,'Price Prediction of Train Dataset')
evaluate(test,test_prep,'Price Prediction of Test Dataset')

st.subheader('Graphical Statistics for Trainset')
st.line_chart(train1.SalePrice,use_container_width=True)
st.bar_chart(train1.YrSold.value_counts())
#print(train.YearBuilt)

def filedownload(data):
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
    return href

st.markdown("""[Train Dataset](filedownload(train1),unsafe_allow_html=True)""")
st.markdown("""[Test Dataset](filedownload(test1),unsafe_allow_html=True)""")

