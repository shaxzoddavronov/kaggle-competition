import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
train=pd.read_csv('train.csv',index_col=0)
test=pd.read_csv('test.csv',index_col=0)





def prepared(train,test):

    corr=train.corrwith(train.SalePrice).abs().sort_values(ascending=False)
    drop_val=corr[corr.values<0.020].index.tolist()
    train=train.drop(drop_val,axis=1)    
    test=test.drop(drop_val,axis=1)

    cols=['LotFrontage','MasVnrArea','GarageYrBlt','MasVnrType','BsmtQual',
        'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2',
        'BsmtUnfSF','TotalBsmtSF','BsmtFullBath','KitchenQual','Functional',
        'GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea',
        'GarageQual','GarageCond','SaleType','Electrical','Exterior1st','Exterior2nd','Utilities','MSZoning']
    for col in cols:
        if train[col].dtype=='float64' or train[col].dtype=='int64':
            train[col]=SimpleImputer(strategy='mean').fit_transform(train[[col]])
            test[col]=SimpleImputer(strategy='mean').fit_transform(test[[col]])
        else:
            train[col]=SimpleImputer(strategy='most_frequent').fit_transform(train[[col]])
            test[col]=SimpleImputer(strategy='most_frequent').fit_transform(test[[col]])
            
    train.dropna(thresh=1460,axis=1,inplace=True)
    test.dropna(thresh=1459,axis=1,inplace=True)

    train_X=train.drop('SalePrice',axis=1)
    train_y=train['SalePrice']
    test_X=test.copy()

    num_pipe=Pipeline([('encoder',OrdinalEncoder()),
                    ('standard',StandardScaler())])
    num_cols=[c for c in train_X.columns if train_X[c].dtype!='object']
    cat_cols=[c for c in train_X.columns if train_X[c].dtype=='object']
    full_pipe=ColumnTransformer([('num',num_pipe,num_cols),
                                ('cat',OrdinalEncoder(),cat_cols)])
    train_prepared=full_pipe.fit_transform(train_X)
    test_prepared=full_pipe.fit_transform(test_X)
    return train_prepared,test_prepared

train_prep,test_prep=prepared(train,test)
print(train_prep.shape)
print(train_prep[1459])