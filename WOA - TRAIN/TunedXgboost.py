from tuned_xgboost import error_coefficient
from tuned_xgboost import WOA1
Wxs = []

Wx = WOA1.WOAXgboost()
Wx.run()
xgb_model = Wx.xgb_model
y_train = Wx.train.get_label()
y_validation = Wx.validation.get_label()
y_test = Wx.test.get_label()
y_data = Wx.data_whole.get_label()
Z_train = xgb_model.predict(Wx.train,ntree_limit=xgb_model.best_ntree_limit)
Z_validation=xgb_model.predict(Wx.validation,ntree_limit=xgb_model.best_ntree_limit)
Z_test = xgb_model.predict(Wx.test,ntree_limit=xgb_model.best_ntree_limit)
Z_data=xgb_model.predict(Wx.data_whole,ntree_limit=xgb_model.best_ntree_limit)
#error_coefficient.print_coefficient(y_train,Z_train)
#error_coefficient.print_coefficient(y_date,Z_date)
#error_coefficient.print_coefficient(y_test,Z_test)
RMSE,MAE,MAPE,R2=error_coefficient.print_coefficient(y_data,Z_data)
RMSE,MAE,MAPE,R2=error_coefficient.print_coefficient(y_train,Z_train)
RMSE,MAE,MAPE,R2=error_coefficient.print_coefficient(y_test,Z_test)
RMSE,MAE,MAPE,R2=error_coefficient.print_coefficient(y_validation,Z_validation)
Wxs.append(Wx)
