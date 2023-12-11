
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


# calculate performance values
def calculate_performance(predictors, target_input, model_input):

    predictions = model_input.predict(predictors)
    
    # - r squared and adj_r_squared
    r_squared     = np.corrcoef(target_input, predictions)[0,1]**2.0
    adj_r_squared = 1 - (1-r_squared)*(len(target_input)-1)/(len(target_input)-predictors.shape[1]-1)
    # ~ print(r_squared)
    # ~ print(adj_r_squared)
    # - rmse and mae
    rmse        = (mean_squared_error(target_input, predictions))**0.5
    mae         = mean_absolute_error(target_input, predictions)
    # ~ print(rmse)
    # ~ print(mae)
    
    # ~ print(predictions, target_input)
    
    return r_squared, adj_r_squared, rmse, mae 

# read dataset
# ~ dataset = pd.read_csv("../datasets/version_20230808/dataset_selected_20230808.csv", sep = ";")
# -- using the new dataset from Jaivime
dataset = pd.read_csv("../datasets/version_20230812/GeospatialModel_20230722.csv")
print(dataset.to_string())

# include pet_p_ratio and dwt_m as predictors
predictors = pd.DataFrame()
# ~ predictors["pet_p_ratio"] = dataset["pet_p_ratio"]
# ~ predictors["dwt_m"]       = dataset["dwt_m"]
# -- using the new dataset from Jaivime
# ~ No,Country,SiteID,SiteUniID,aet_p,pet_p,dwt,peak_v,cxtfit_v,cxtfit_D,cxtfit_R2,Applicability_tau
predictors["pet_p_ratio"] = dataset["pet_p"]
predictors["dwt_m"]       = dataset["dwt"]

# add a multiplicative term incorporating the above pre
# - calculate the average values of predictor variables
avg_pet_p_ratio = predictors["pet_p_ratio"].mean()
avg_dwt_m       = predictors["dwt_m"].mean()
print("avg_pet_p_ratio and avg_dwt_m")
print(avg_pet_p_ratio)
print(avg_dwt_m)
print("")
# ~ avg_pet_p_ratio and avg_dwt_m
# ~ 2.3613636363636363
# ~ 116.14999999999999
#
# - using the average values from Jaivime (see the R code, note these values are very much the same as above)
avg_pet_p_ratio = 2.36136
avg_dwt_m       = 116.15
# - using the centered values
multiplicative_term = (predictors["pet_p_ratio"] - avg_pet_p_ratio) * (predictors["dwt_m"] - avg_dwt_m)
predictors["multiplicative_term"] = multiplicative_term
# ~ print(multiplicative_term)

# target variables
# ~ target = dataset["Applicability_tau_yr"].astype(float)
# -- using the new dataset from Jaivime
# ~ No,Country,SiteID,SiteUniID,aet_p,pet_p,dwt,peak_v,cxtfit_v,cxtfit_D,cxtfit_R2,Applicability_tau
target = dataset["Applicability_tau"].astype(float)
# ~ print(target)

# fit the model using all data
mlr_model = LinearRegression()
mlr_model.fit(predictors, target)

# intercept and regression coefficients
print("intercept and regression coefficients (using all data)")
print(mlr_model.intercept_)
print(mlr_model.coef_)
# ~ print(type(mlr_model.intercept_))
# ~ print(type(mlr_model.coef_))
print("")
# ~ intercept and regression coefficients (using all data)
# ~ -17.824228930512973
# ~ [ 3.91379777  0.36500297 -0.08373464]

# get performance values
r_squared_all, adj_r_squared_all, rmse_all, mae_all = calculate_performance(predictors, target, mlr_model) 
print("r_squared_all, adj_r_squared_all, rmse_all, mae_all")   
print(r_squared_all, adj_r_squared_all, rmse_all, mae_all)   
print("")


# cross validation
#
# ~ # - using RepeatedKFold (see e.g. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html#sklearn.model_selection.RepeatedKFold)
# ~ cross_validation_method = RepeatedKFold(n_splits = 5, n_repeats = 1000, random_state = 20230823)
# ~ cross_validation_method = RepeatedKFold(n_splits = 5, n_repeats = 20000, random_state = 20230823)
#
# - cross validation - leave one out, using LeavePOut with p = 1
cross_validation_method = LeavePOut(1)


# make splits
cross_validation_method.get_n_splits(predictors, target)

# table/data frame for storing the cross validation result
cross_val_df = pd.DataFrame(\
                             {
                              'i'                   : pd.Series(dtype='int'),
                              'intercept'           : pd.Series(dtype='float'),
                              'reg_coef_1'          : pd.Series(dtype='float'),
                              'reg_coef_2'          : pd.Series(dtype='float'),
                              'reg_coef_3'          : pd.Series(dtype='float'),
                              'r_squared_train'     : pd.Series(dtype='float'),
                              'adj_r_squared_train' : pd.Series(dtype='float'),
                              'rmse_train'          : pd.Series(dtype='float'),
                              'mae_train'           : pd.Series(dtype='float'),
                              'r_squared_test'      : pd.Series(dtype='float'),
                              'adj_r_squared_test'  : pd.Series(dtype='float'),
                              'rmse_test'           : pd.Series(dtype='float'),
                              'mae_test'            : pd.Series(dtype='float'),
                             })


# perform cross validation
i = 0
for train_index, test_index in cross_validation_method.split(predictors, target): 
# ~ for fold, (train_index, test_index) in enumerate(cross_validation_method.split(predictors, target)):

   i = i + 1
   print(i)

   # ~ print("TRAIN:", train_index, "TEST:", test_index)

   # train dataset
   predictors_train = None
   predictors_train = pd.DataFrame()
   predictors_train["pet_p_ratio"]         = predictors["pet_p_ratio"][train_index]
   predictors_train["dwt_m"]               = predictors["dwt_m"][train_index]
   predictors_train["multiplicative_term"] = predictors["multiplicative_term"][train_index]
   target_train     = None
   target_train                            = target[train_index]               
   
   # test dataset
   predictors_test = None
   predictors_test = pd.DataFrame()
   predictors_test["pet_p_ratio"]          = predictors["pet_p_ratio"][test_index]
   predictors_test["dwt_m"]                = predictors["dwt_m"][test_index]
   predictors_test["multiplicative_term"]  = predictors["multiplicative_term"][test_index]
   target_test     = None
   target_test                             = target[test_index]               

   # fit the model using the train dataset
   mlr_model_train = None
   del mlr_model_train
   mlr_model_train = LinearRegression()
   mlr_model_train.fit(predictors_train, target_train)
   intr_train = mlr_model_train.intercept_
   coef_train = mlr_model_train.coef_
   
   # ~ print(intr_train)
   # ~ print(coef_train)

   # get the performance based on the train data
   r_squared_train, adj_r_squared_train, rmse_train, mae_train = calculate_performance(predictors_train, target_train, mlr_model_train)
   # ~ print(r_squared_train, adj_r_squared_train, rmse_train, mae_train)    

   # get the performance based on the test data
   r_squared_test, adj_r_squared_test, rmse_test, mae_test = calculate_performance(predictors_test, target_test, mlr_model_train)    
   # ~ print(r_squared_test, adj_r_squared_test, rmse_test, mae_test)    
   
   # add the result to the data frame
   new_row = None
   del new_row
   new_row = {
               'i'                   : i,
               'intercept'           : intr_train,
               'reg_coef_1'          : coef_train[0],
               'reg_coef_2'          : coef_train[1],
               'reg_coef_3'          : coef_train[2],
               'r_squared_train'     : r_squared_train,
               'adj_r_squared_train' : adj_r_squared_train,
               'rmse_train'          : rmse_train,
               'mae_train'           : mae_train,
               'r_squared_test'      : r_squared_test,
               'adj_r_squared_test'  : adj_r_squared_test,
               'rmse_test'           : rmse_test,
               'mae_test'            : mae_test
              }
   print(new_row)
   cross_val_df.loc[len(cross_val_df)] = new_row                         
                             
   # ~ if i == 1: break

# write data frame to a csv file
# ~ cross_val_df.to_csv("cv_result_kfold_v20230823_1000.csv" , index = False)  
# ~ cross_val_df.to_csv("cv_result_kfold_v20230823_20000.csv", index = False)  
cross_val_df.to_csv("cv_result_leave_one_out_v20230824.csv"  , index = False)  

# TODO: plot the histogram (see e.g. https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.hist.html)
