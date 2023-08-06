import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import plotly.figure_factory as ff

st.image('Titanic.png',width=500)
st.title('Titanic Survival Prediction')

st.write("""On April 15, 1912, during her maiden voyage, the widely considered “unsinkable”
          RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough 
         lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.""")

st.write("""While there was some element of luck involved in surviving, it seems some groups of people were
          more likely to survive than others.""")

st.subheader('Here is train set without any preporcessing')
st.write('The goal is predicting passenger survived or not')
   
st.write('Original data')
data=pd.read_csv('train_titanic.csv')
test=pd.read_csv('test_titanic.csv')
gen_types=data['Sex'].unique()
gen=st.sidebar.selectbox('Genders',gen_types)
embark_types=data[data['Embarked'].notnull()].Embarked.unique()
emb=st.sidebar.selectbox('Embarked (C=Cherbourg,Q=Queenstown,S=Southampton)',embark_types)
age=st.sidebar.slider('Max Age',0,int(data['Age'].max()),int(data['Age'].max())//2)
fare=st.sidebar.slider('Max Fare',0,int(data['Fare'].max()),int(data['Fare'].max())//2)
parch_types=np.sort(data['Parch'].unique())
parch=st.sidebar.multiselect('Parch',parch_types,parch_types)
sibsp_types=np.sort(data['SibSp'].unique())
sibsp=st.sidebar.multiselect('Parch',sibsp_types,sibsp_types)
data=data[(data.Sex==gen)&(data.Embarked==emb)&(data.Age<=age)&(data.Fare<=fare)&(data.Parch.isin(parch))&(data.SibSp.isin(sibsp))]
data.index=np.arange(len(data))

st.dataframe(data)
def show_data(df,title):    
    st.write(title)
    data=pd.read_csv(df)
    st.dataframe(data)

show_data('future_engineering.csv','Sample data used feature engineering')
show_data('sample_filtered_data.csv','Sample filtered data')
show_data('prepared_data.csv','Sample prepared data')

index=st.number_input('Insert index',0,len(data)-1)
labels=pd.read_csv('labels.csv')
st.write('Prediction (Survived:1 ,Not survived:0)')
st.success(labels.loc[index]['Survived'])

st.subheader('Line plot for filtered trainset')
st.line_chart(data.Fare,use_container_width=True)
st.line_chart(data[['Parch','SibSp']],use_container_width=True)

original_data=pd.read_csv('train_titanic.csv')
embark_types=['S','C','Q']
ages=[]
for embark in embark_types:
    emb_age=original_data[original_data.Embarked==embark].Fare.values
    ages.append(emb_age)
fig=ff.create_distplot(ages,embark_types,bin_size=[.1, .25, .5])
st.plotly_chart(fig)    
    
st.subheader('Bar plots for original Trainset')
fig, ax = plt.subplots(1,3,figsize=(16, 6))
sns.countplot(data=original_data,x='Sex',hue='Survived',palette='Set1',ax=ax[0])
sns.countplot(data=original_data,x='Pclass',hue='Survived',palette='Set1',ax=ax[1])
sns.countplot(data=original_data,x='Embarked',hue='Survived',palette='Set1',ax=ax[2])
plt.tight_layout()
st.pyplot(fig)

def filedownload(data):
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
    return href

st.markdown("""[Train Dataset](filedownload(original_data),unsafe_allow_html=True)""")
st.markdown("""[Test Dataset](filedownload(test),unsafe_allow_html=True)""")