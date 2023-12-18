
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut

from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

# calculate performance values
def calculate_performance(predictors, target_input, model_input):

    predictions = model_input.predict(predictors)
    
    # - r squared and adj_r_squared
    if len(target_input) > 1:
        r_squared     = np.corrcoef(target_input, predictions)[0,1]**2.0
        adj_r_squared = 1 - (1-r_squared)*(len(target_input)-1)/(len(target_input)-predictors.shape[1]-1)
    else:
        r_squared     = -9999
        adj_r_squared = -9999
    
    # - rmse and mae
    if len(target_input) > 1:
        rmse        = (mean_squared_error(target_input, predictions))**0.5
        mae         = mean_absolute_error(target_input, predictions)
    else:
        rmse = ((predictions - target_input)**2.0)**0.5
        mae  = np.absolute(predictions - target_input)
    
    
    # ~ print(rmse)
    # ~ print(mae)
    
    # ~ print(predictions, target_input)
    
    return r_squared, adj_r_squared, rmse, mae 




# read dataset
dataset = pd.read_csv("example_uk.csv", encoding='ISO-8859-1', delimiter=',')
print(dataset.to_string())
# - replace 1.00E+31 with NaN
dataset.replace(1.00E+31, np.nan, inplace=True)
# - drop all rows with NaN
dataset = dataset.dropna()
# - reset index
dataset = dataset.reset_index(drop = True)

# define the target variable
#~ target = dataset["Species normalized"].astype(float)
target = dataset["Species"].astype(float)





# define the predictors
# - starting with an empty dataframe
predictors = pd.DataFrame()
# - using the following variables Area, Groundwater recharge,Groundwater depth,Evaporation,Discharge,Salinity,BOD,TP,NOXN,bod,ec
predictors["Area"]                 = dataset["Area"]
predictors["Groundwater recharge"] = dataset["Groundwater recharge"]
predictors["Groundwater depth"]    = dataset["Groundwater depth"]
predictors["Evaporation"]          = dataset["Evaporation"]
predictors["Discharge"]            = dataset["Discharge"]
predictors["Salinity"]             = dataset["Salinity"]
predictors["BOD"]                  = dataset["BOD"]
predictors["TP"]                   = dataset["TP"]
predictors["NOXN"]                 = dataset["NOXN"]
predictors["bod"]                  = dataset["bod"]
predictors["ec"]                   = dataset["ec"]


# fit the model using all data
mlr_model = LinearRegression()
mlr_model.fit(predictors, target)


# intercept and regression coefficients
print("intercept and regression coefficients (using all data)")
print(mlr_model.intercept_)
print(mlr_model.coef_)


# get performance values
r_squared_all, adj_r_squared_all, rmse_all, mae_all = calculate_performance(predictors, target, mlr_model) 
print("r_squared_all, adj_r_squared_all, rmse_all, mae_all")   
print(r_squared_all, adj_r_squared_all, rmse_all, mae_all)   
print("")


# empty table/data frame for storing the the result from the model 
result_df_empty = pd.DataFrame(\
                             {
                              'i'                   : pd.Series(dtype='int'),
                              'intercept'           : pd.Series(dtype='float'),
                              'reg_coef_1'          : pd.Series(dtype='float'),
                              'reg_coef_2'          : pd.Series(dtype='float'),
                              'reg_coef_3'          : pd.Series(dtype='float'),
                              'reg_coef_4'          : pd.Series(dtype='float'),
                              'reg_coef_5'          : pd.Series(dtype='float'),
                              'reg_coef_6'          : pd.Series(dtype='float'),
                              'reg_coef_7'          : pd.Series(dtype='float'),
                              'reg_coef_8'          : pd.Series(dtype='float'),
                              'reg_coef_9'          : pd.Series(dtype='float'),
                              'reg_coef_10'         : pd.Series(dtype='float'),
                              'reg_coef_11'         : pd.Series(dtype='float'),
                              'r_squared_train'     : pd.Series(dtype='float'),
                              'adj_r_squared_train' : pd.Series(dtype='float'),
                              'rmse_train'          : pd.Series(dtype='float'),
                              'mae_train'           : pd.Series(dtype='float'),
                              'r_squared_test'      : pd.Series(dtype='float'),
                              'adj_r_squared_test'  : pd.Series(dtype='float'),
                              'rmse_test'           : pd.Series(dtype='float'),
                              'mae_test'            : pd.Series(dtype='float'),
                             })

# result from the model fitted using all data
result_df_all = result_df_empty
new_row = None
del new_row
new_row = {
            'i'                   : -1,
            'intercept'           : mlr_model.intercept_,
            'reg_coef_1'          : mlr_model.coef_[0],
            'reg_coef_2'          : mlr_model.coef_[1],
            'reg_coef_3'          : mlr_model.coef_[2],
            'reg_coef_4'          : mlr_model.coef_[3],
            'reg_coef_5'          : mlr_model.coef_[4],
            'reg_coef_6'          : mlr_model.coef_[5],
            'reg_coef_7'          : mlr_model.coef_[6],
            'reg_coef_8'          : mlr_model.coef_[7],
            'reg_coef_9'          : mlr_model.coef_[8],
            'reg_coef_10'         : mlr_model.coef_[9],
            'reg_coef_11'         : mlr_model.coef_[10],
            'r_squared_train'     : r_squared_all,
            'adj_r_squared_train' : adj_r_squared_all,
            'rmse_train'          : rmse_all,
            'mae_train'           : mae_all,
            'r_squared_test'      : -9999,
            'adj_r_squared_test'  : -9999,
            'rmse_test'           : -9999,
            'mae_test'            : -9999
           }
print(new_row)
result_df_all.loc[len(result_df_all)] = new_row
# write data frame to a csv file
result_df_all.to_csv("cv_result_all_example_uk.csv"  , index = False)  


print(mlr_model.summary())
                         
                             
