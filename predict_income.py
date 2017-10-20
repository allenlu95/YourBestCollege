import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

#function creates a debt column by joining two distinct columns
def create_debt(df):
    df['C150_4'].fillna(df['C150_L4'], inplace=True)
    del df['C150_L4']
    return df
#Function gets us the columns we want to use for our model
def get_columns(df):
    df = df[(df['MD_EARN_WNE_P10'].isnull()) & (df['LOAN_EVER']>0) & (df['TUITFTE']>0)]
    df = df.loc[:,df.isnull().sum()<100]
    return df.iloc[:,244:].columns
#Function makes sure there's not too many nulls
def select_columns(df):
    df = df[df['MD_EARN_WNE_P10'].notnull()]
    return df.loc[:,df.isnull().sum()<2000]
#Gets our numeric values
# TODO: create categorical variables
def get_numeric(df):
    pd.to_numeric(df)
    return df
#Function to get mean log error
def mean_squared_log_error(y_true, y_pred):
    log_diff = np.log(y_pred+1) - np.log(y_true+1)
    return np.sqrt(np.mean(log_diff**2))


if __name__=='__main__':
#Cleaning and reading in our data
    df = (pd.read_csv('MERGED2011_12_PP.csv')
          .pipe(create_debt)
          .replace('PrivacySuppressed',np.nan)
          .pipe(select_columns)
          .convert_objects(convert_numeric=True)
        .iloc[:,7:])
#Split into testing/training
    mask = get_columns(df)
    y = df['MD_EARN_WNE_P10']
    X = df[mask]._get_numeric_data()
    X_train, X_test, y_train, y_test = train_test_split(X.drop('MD_EARN_WNE_P10',axis=1),y)
#Get baseline model
    median_list = [y_test.median()]*len(y_test)
    print 'Baseline score {}'.format(mean_squared_log_error(y_test, np.array(median_list)))
#Modeling
    p = Pipeline([('I', Imputer()), ('rf', RandomForestRegressor())])
    params =  dict(I__strategy=['mean','median','most_frequent'],
    rf__n_estimators = [50, 100, 200],rf__max_features = ["sqrt","log2"])
    gscv = GridSearchCV(p, params, scoring = make_scorer(mean_squared_log_error, greater_is_better=False), cv = 5, n_jobs =-1)
    gscv.fit(X_train, y_train)
    print 'Validation score {}'.format(abs(gscv.best_score_))
# Predicting on test set
    X_train = gscv.best_estimator_.named_steps['I'].fit_transform(X_train)
    rf_best = gscv.best_estimator_.named_steps['rf'].fit(X_train, y_train)
    X_test = gscv.best_estimator_.named_steps['I'].fit_transform(X_test)
    print 'Test score {}'.format(mean_squared_log_error(rf_best.predict(X_test),y_test))
