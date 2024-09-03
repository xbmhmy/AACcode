from tuned_xgboost import error_coefficient
from tuned_xgboost import bas_xgboost
bxs=[]

bx=bas_xgboost.BasXgboost()
bx.run()
xgb_model=bx.xgb_model
y_train=bx.train.get_label()
y_validation=bx.validation.get_label()
y_test=bx.test.get_label()
y_data=bx.data_whole.get_label()
Z_train = xgb_model.predict(bx.train,ntree_limit=xgb_model.best_ntree_limit)
Z_validation=xgb_model.predict(bx.validation,ntree_limit=xgb_model.best_ntree_limit)
Z_test = xgb_model.predict(bx.test,ntree_limit=xgb_model.best_ntree_limit)
Z_data=xgb_model.predict(bx.data_whole,ntree_limit=xgb_model.best_ntree_limit)
#error_coefficient.print_coefficient(y_train,Z_train)
#error_coefficient.print_coefficient(y_date,Z_date)
#error_coefficient.print_coefficient(y_test,Z_test)
RMSE,MAE,MAPE,R2=error_coefficient.print_coefficient(y_data,Z_data)
RMSE,MAE,MAPE,R2=error_coefficient.print_coefficient(y_train,Z_train)
RMSE,MAE,MAPE,R2=error_coefficient.print_coefficient(y_test,Z_test)
RMSE,MAE,MAPE,R2=error_coefficient.print_coefficient(y_validation,Z_validation)

bxs.append(bx)
