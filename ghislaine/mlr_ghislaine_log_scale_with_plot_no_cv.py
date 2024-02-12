
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

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor 

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
    
    
    return r_squared, adj_r_squared, rmse, mae 


# read dataset
# ~ # - england
# ~ dataset = pd.read_csv("example_uk.csv", encoding='ISO-8859-1', delimiter=',')
# - australia
dataset = pd.read_csv("wetland_species_aus.csv", encoding='ISO-8859-1', delimiter=',')

# ~ print(dataset.to_string())

# - replace 1.00E+31 with NaN
dataset.replace(1.00E+31, np.nan, inplace=True)
# - drop all rows with NaN
dataset = dataset.dropna()
# - reset index
dataset = dataset.reset_index(drop = True)


##########################################################################
# DO THE FOLLOWING FOR TRANSFORMING DATASET VALUES TO THEIR LOG VALUES

# remove some columns that are not needed 
dataset = dataset.drop(['Wetlands name', 'State', "lat", "lon"], axis = 1)

# remove all negative values and zero 
dataset = dataset.mask(dataset <= 0)
dataset = dataset.dropna()
# - reset index
dataset = dataset.reset_index(drop = True)

# convert the dataset values to log scales
dataset = np.log(dataset)

##########################################################################


print(dataset)


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



# fit the model using all data - using sklearn
mlr_model = LinearRegression()
mlr_model.fit(predictors,
 target)

# fit the model using all data - using ols
reg_ols = sm.OLS(target, sm.add_constant(predictors)).fit()
print("")
print("reg ols summary")
print(reg_ols.summary())
print("")

# intercept and regression coefficients
print("")
print("intercept and regression coefficients from sklearn (using all data)")
print(mlr_model.intercept_)
print(mlr_model.coef_)
print("")

# VIF
vif_data = pd.DataFrame() 
vif_data["feature"] = predictors.columns 
  
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(predictors.values, i) 
                          for i in range(len(predictors.columns))] 
  
print(vif_data)


# get the predicted value
predictions = pd.DataFrame(mlr_model.predict(predictors))
print(predictions)


# create scatter plots between 'measurements (y)' and 'predicted values (x)'
plt.figure(figsize=(8, 6))
plt.scatter(x = target, y = predictions)
plt.title('Scatter Plot: Predictions vs Measurements')

# add the trendline to the plot
model_linear_plot = LinearRegression()

print(predictions)
print(target)

model_linear_plot.fit(predictions, target)
plt.plot(predictions, model_linear_plot.predict(predictions), color='red', linewidth=2)
plt.savefig('mlr_result.png')
plt.show()
