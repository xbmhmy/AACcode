import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
def data_processing(dataset1 = r'2.3.xlsx'):
    dataset = pd.read_excel(dataset1,engine = 'openpyxl')
    X_index = dataset.columns[:-1]
    y_index = dataset.columns[-1]
    X = dataset.loc[:, dataset.columns != 'fc']
    y = dataset.loc[:, 'fc']
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_train, X_test_validation, y_train, y_test_validation = train_test_split(X, y, test_size = 0.40, random_state = 10)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test_validation, y_test_validation, test_size = 0.50, random_state = 10)
    return dataset, X_index, y_index, X_train, y_train, X_test, y_test, X_validation, y_validation

def dg(dataset1 = r'2.3.xlsx'):
    dataset = pd.read_excel(dataset1,engine = 'openpyxl')
    X_index = dataset.columns[:-1]
    y_index = dataset.columns[-1]
    X = dataset.loc[:, dataset.columns != 'fc']
    y = dataset.loc[:, 'fc']
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_train, X_test_validation, y_train, y_test_validation = train_test_split(X, y, test_size = 0.40, random_state = 10)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test_validation, y_test_validation, test_size = 0.50, random_state = 10)
    return X,y

def get_set():
    dataset, X_index, y_index, X_train, y_train, X_test, y_test, X_validation, y_validation = data_processing()
    X,y=dg()
    data_whole = xgb.DMatrix(data = X, label = y)
    train = xgb.DMatrix(data = X_train,label = y_train)
    validation = xgb.DMatrix(data = X_validation,label = y_validation)
    test = xgb.DMatrix(data = X_test,label = y_test)
    return data_whole,train,validation,test
