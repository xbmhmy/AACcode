from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
import numpy as np
import pandas as pd

def get_MAPE(Y,predicted_Y):
    y_true, y_pred = np.array(Y), np.array(predicted_Y)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def predict(xgb_model,train,validation,test):
    z_train = xgb_model.predict(train,ntree_limit=xgb_model.best_ntree_limit)
    z_validation=xgb_model.predict(validation,ntree_limit=xgb_model.best_ntree_limit)
    z_test = xgb_model.predict(test,ntree_limit=xgb_model.best_ntree_limit)


def print_coefficient(labels,predicts):

    RMSE = np.sqrt(mean_squared_error(labels,predicts))
    print('RMSE of model on the  data set: {0:6.2f}'.format(RMSE))
    MAE = mean_absolute_error(labels,predicts)
    print('MAE of XGB model on the  data set: {0:6.2f}'.format(MAE))
    MAPE = get_MAPE(labels,predicts)
    print('MAPE of XGB model on the data set: {0:6.2f}'.format(MAPE))
    R2 = r2_score(labels,predicts)
    print('R2 of XGB model on the  data set: {0:6.2f}'.format(R2))

    return RMSE,MAE,MAPE,R2
