import numpy as np
import xgboost as xgb
from tuned_xgboost import data_processing
np.random.seed(0)

def get_trained_model(**kwargs):
    dataset, X_index, y_index, X_train, y_train, X_test, y_test, X_validation, y_validation = data_processing.data_processing()
    param = {
        'eta': 0.13573895,'max_depth': 10,'gamma':0,'subsample': 0.57899197,'alpha':0,'reg_lambda':31.56958254,'objective': 'reg:squarederror','eval_metric': 'rmse','nthread': 4,'verbosity':0
        }
    
    train = kwargs['train']
    test = kwargs['test'] 
    

    evallist = [(train, 'train'),(test, 'test')]
    evals_result = {}
    
    xgb_model = xgb.train(param,train, num_boost_round=2000, evals=evallist, evals_result=evals_result, early_stopping_rounds=50)
    
    
    return xgb_model, evals_result
