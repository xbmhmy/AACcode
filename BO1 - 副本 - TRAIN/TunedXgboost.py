from tuned_xgboost import error_coefficient
from tuned_xgboost import BO1
Bxs = []


Bx =  BO1.BOXgboost()
Bx.run()
xgb_model = Bx.xgb_model
y_train = Bx.train.get_label()
y_validation = Bx.validation.get_label()
y_test = Bx.test.get_label()
y_data = Bx.data_whole.get_label()
Z_train = xgb_model.predict(Bx.train,ntree_limit=xgb_model.best_ntree_limit)
Z_validation=xgb_model.predict(Bx.validation,ntree_limit=xgb_model.best_ntree_limit)
Z_test = xgb_model.predict(Bx.test,ntree_limit=xgb_model.best_ntree_limit)
Z_data=xgb_model.predict(Bx.data_whole,ntree_limit=xgb_model.best_ntree_limit)
#error_coefficient.print_coefficient(y_train,Z_train)
#error_coefficient.print_coefficient(y_date,Z_date)
#error_coefficient.print_coefficient(y_test,Z_test)
RMSE,MAE,MAPE,R2=error_coefficient.print_coefficient(y_data,Z_data)
RMSE,MAE,MAPE,R2=error_coefficient.print_coefficient(y_train,Z_train)
RMSE,MAE,MAPE,R2=error_coefficient.print_coefficient(y_test,Z_test)
RMSE,MAE,MAPE,R2=error_coefficient.print_coefficient(y_validation,Z_validation)
Bxs.append(Bx)
