import streamlit as st 
import pandas as pd 
import numpy as np 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn  import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Union

from pycaret.datasets import get_data



st.markdown(f"<h2 style='text-align: center; color: red;'> Pycaret Project📜🔍</h2>", unsafe_allow_html=True)
# st.header("Pycaret Project ")
st.sidebar.subheader("**About The App 🤩👋**")
st.sidebar.markdown("🔍⚒️ An interactive tool powered by PyCaret for quick and efficient machine learning model building and analysis.")
Select = st.sidebar.selectbox("Select Option",('Package','Show code & Other Resources'))
if Select=='Package':
    data_set=st.file_uploader('Upload File',type=['csv','txt','xlsx'])
       
    if data_set is not None:
            df=pd.read_csv(data_set)

#  1- automate your preprocessing , detect columns types , null values 

            col=df.columns

            list_missing=df.isna().sum()
                    
            i=0 
            list_of_missing=[]


            for i in range(len(list_missing)):
                if list_missing.iloc[i]!=0:
                    list_of_missing.append(df.columns[i])
                i+=1 


            st.subheader('**Data Details**')  
            st.write('The shape of data : ',df.shape)
                
            st.subheader('**Columns of the dataset**')  
            st.write(df.columns.to_list())

            st.subheader('Data Types Of Columns')
            d_type=df.dtypes
            st.write(d_type) 


            st.subheader('Check missing values')
            list_missing=df.isna().sum()
            st.write(list_missing)
            i=0 
            list_of_missing=[]
            

            for i in range(len(list_missing)):
                if list_missing.iloc[i]!=0:
                    list_of_missing.append(df.columns[i])
                i+=1 
            if list_of_missing!=[] :    
                st.write('columns with missing values : ',list_of_missing)
                 


 # 2- ask user to decide what columns to drop and what column to predict ONLY
            target_col = st.selectbox("Select Target variable",col)
            num_column=df.select_dtypes (include=['int64', 'float64']).columns
         
            d_type=df.dtypes
            num_feature=[]
            cat_feature=[]
            for j in range(len(d_type)):
                if d_type.iloc[j]=='object':
                    cat_feature.append(df.columns[j])
                    
                elif d_type.iloc[j]=='float64'or d_type.iloc[j]=='int64' :
                    num_feature.append(df.columns[j]) 
                    
            # st.write('num_feature',num_feature)
            # st.write('cat_feature',cat_feature) 
            # st.write(df[target_col])   
        #    df[target_col] == df[num_feature][target_col])
            # st.write(target_col in num_feature)

            if target_col in num_feature:
                alg='Regression'              
                
            else:
                
                alg='classification'

            # st.write(alg) 
            st.markdown(f"<h2 style='text-align: center; color: teal;'>{alg}</h2>", unsafe_allow_html=True)



            # Encoding for categorical features
            le=LabelEncoder()
            for i in range(len(cat_feature)):
                df[cat_feature[i]]=le.fit_transform(df[cat_feature[i]])
                i+=1
            # st.write(df[cat_feature]) 

# 4- ask user what techniques he want to apply in the columns , 
# ask him like what do you want to do with categorical ( most frequent or just put additional class for missing value ) 
# and ask him again for continuous ( mean or median or mode ) and apply what he choose to your columns depends on every column type 
            
               
            mean_impute=SimpleImputer(strategy='mean',missing_values=np.nan)       
            mode_impute=SimpleImputer(strategy='most_frequent',missing_values=np.nan) 
            median_impute=SimpleImputer(strategy='median',missing_values=np.nan)
            constant_impute=SimpleImputer(strategy='constant',missing_values=np.nan,fill_value=10)


            # st.write(' Do you want to fill missing values  numerical features ')
            q_num=st.radio(' Do you want to fill missing values  numerical features ?',options=['Yes','No'])
            if q_num=='Yes':
                method=st.radio(' which method you want to apply for numerical features?',options=['mean','median','most_frequent'])
                if method=='mean':
                    i=0
                    j=0   
                    for i in range(len(list_of_missing)):
                        if list_of_missing[i]==num_feature[i]:
                            for j in  range(df[num_feature].shape[0]):
                                    df[num_feature[i]]=mean_impute.fit_transform(df[num_feature[i]].values.reshape(-1,1))
                                    j+=1 
                elif method=='median':
                    i=0
                    j=0   
                    for i in range(len(list_of_missing)):
                        if list_of_missing[i]==num_feature[i]:
                            for j in  range(df[num_feature].shape[0]):
                                    df[num_feature[i]]=median_impute.fit_transform(df[num_feature[i]].values.reshape(-1,1))
                                    j+=1                           
                elif method=='most_frequent':
                        i=0
                        j=0   
                        for i in range(len(list_of_missing)):
                            if list_of_missing[i]==num_feature[i]:
                                for j in  range(df[num_feature].shape[0]):
                                        df[num_feature[i]]=mode_impute.fit_transform(df[num_feature[i]].values.reshape(-1,1))
                                        j+=1         
                     
                # st.write(df[num_feature].iloc[:,0:3])    
            
            q_cat=st.radio(' Do you want to fill missing values  categorical features ?',options=['Yes','No'])
            if q_cat=='Yes': 
                    method=st.radio(' which method you want to apply for categorical features ?',options=['most_frequent','additional class:constant']) 
                    if method=='most_frequent':
                        for i in range(len(list_of_missing)):
                            if   list_of_missing[i]==cat_feature[i]:
                                for j in  range(df[list_of_missing].shape[0]):
                                    df[list_of_missing[i]]=mode_impute.fit_transform(df[list_of_missing[i]].values.reshape(-1, 1))[:,0]
                                    j+=1

                                i+=1 

                    elif method=='additional class : constant':
                         for i in range(len(list_of_missing)):
                            if   list_of_missing[i]==cat_feature[i]:
                                for j in  range(df[list_of_missing].shape[0]):
                                    df[list_of_missing[i]]=constant_impute.fit_transform(df[list_of_missing[i]].values.reshape(-1, 1))[:,0]
                                    j+=1

                                i+=1 
                        
                         
                                    
                    # st.write(df[cat_feature]  ) 

            drop_col=st.multiselect('Choose the coulmn to drop if you want ',options=df.columns)
            if drop_col:
                df=df.drop(drop_col,axis=1)
                 
            
            x=df
            x=df.drop(target_col,axis=1)
            y=df[target_col]
            st.subheader('Data after preprocessing')
            st.write(x)
            # st.write(y)   
            if alg=='classification':
                from pycaret.classification import *
                # btn1=st.button('classification by pycaret')
                # if btn1:
                st.subheader('classification by pycaret')
                    # Initialize 
                clf = setup(data = df,target=y,session_id = 123)
                st.subheader('Initialize')
                st.dataframe(pull())
                # train
                st.markdown(f"<h2 style='text-align: center; color: orange;'>Pycaret Report</h2>", unsafe_allow_html=True)

                comp_model= st.multiselect("Select models to compare report",['lr', 'dt', 'rf','et','knn','ridge','svm','dummy'])
                if comp_model:
                    best = compare_models(include=comp_model)  
                    # save_model(best,model_name='best model') 
                    st.subheader('Train')
                    st.subheader('The pycaret model compare report')
                    st.dataframe(pull())
                    # create model
                    st.subheader('create best model')
                    b = create_model(best) 
                    st.dataframe(pull())

                    # plot
                    # plot_model(b, plot='feature', display_format='streamlit')

                    # predict 
                    pred=predict_model(b)
                    st.subheader('predict model')
                    st.dataframe(pred)
                    
                    # save model
                    save_model(b, 'my_best model')

            elif alg=='Regression':
                from pycaret.regression import *
                st.subheader('Regression by pycaret')
                # Initialize 
                clf = setup(data = df,target=y,session_id = 123)
                st.subheader('Initialize')
                st.dataframe(pull())
                # train
                comp_model= st.multiselect("Select models to compare report",['lr', 'dt', 'rf','et','knn','ridge','svm','dummy'])
                if comp_model:
                
                    best = compare_models(comp_model)  
                    # save_model(best,model_name='best model') 
                    st.subheader('Train')
                    st.write('the pycaret model compare report')
                    st.dataframe(pull())
                    # create model
                    st.subheader('create best model')
                    b = create_model(best) 
                    st.dataframe(pull())
                    

                    # plot
                    # plot_model(b, plot='feature', display_format='streamlit')
                    

                    # predict 
                    pred=predict_model(b)
                    st.subheader('predict model')
                    st.dataframe(pred)

                    # save model
                    save_model(b, 'my_best model')


if Select=='Show code & Other Resources':
    
    code=""" 
you can see the code of the app on my github account
https://github.com/AyaAttia20/Pycaret_App

other Resources:
https://pycaret.org/
https://streamlit.io/
"""


    st.code(code, language='python')
    st.write("Created By: Aya ")





               
                
