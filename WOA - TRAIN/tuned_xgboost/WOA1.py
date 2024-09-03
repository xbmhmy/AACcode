import numpy as np
import xgboost as xgb

from . import data_processing as dp
from . import set_parameters as sp

class WOAXgboost():
    
    def __init__(self):
        data_whole,train,validation,test=dp.get_set()
        self.train=train
        self.validation=validation
        self.test=test
        self.data_whole=data_whole
        
    def run(self):

        self.xgb_model,self.evals_result=sp.get_trained_model(train=self.train,
                                                              test=self.test)
