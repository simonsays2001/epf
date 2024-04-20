# -*- coding: utf-8 -*-
'''
Created on Mon Oct 17 09:18:51 2022

@author: simon

- This file creates the Expert/AR model for both 2021 and 2022 and puts
- the forecast inside the dictionary "forecast_dict". It takes some time to run
- but instead of running the file it is also possible to uncomment the two
- first lines and load the forecats directly from the previously saved file.
'''

forecast_dict['expert_2021']  = pd.read_pickle(f"{path}\expert_forecast_21_short")
forecast_dict['expert_2022']  = pd.read_pickle(f"{path}\expert_forecast_22_short")


import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation
import statsmodels.api as sm

path = r"C:\Users\simon\OneDrive\Documents\Skola\HT22\EPF_project\data"
df = pd.read_pickle(f"{path}\df")
df.set_index('date', inplace=True)

# Define function to transform data using MAD-ASINH.
def transform(d, i, j):
    global c
    global a
    global b
    a = np.median(d.iloc[(168+8760)+(i*24):j+(i*24)])
    b = (d.iloc[(168+8760)+(i*24):j+(i*24)
         ].apply(median_abs_deviation)*1.48260221851)
    b = b[0].tolist()
    c = np.arcsinh((d-a)/b)
# Define function to inverse transformation   
def inverse(d):
    global c
    c = np.sinh(d)*b+a
###############################################################################
# Create some lists which will be used in the Expert model.
predict_list = (['predict_1', 'predict_2', 'predict_3', 'predict_4'])
df = df[['Price_REG1', 'Price_REG2', 'Price_REG3', 'Price_REG4']]
day_list = list()
for i in range(1, 8):
    day_list.append(f'day_{i}')  
x_list = []
for i in range(1,5):
    a = [f'Price_24_{i}', f'Price_48_{i}', f'Price_168_{i}', f'Price_{i}_min', 
     f'Price_{i}_max', f'Price_last_{i}', 'day_1', 'day_2', 'day_3', 'day_4', 
     'day_5', 'day_6', 'day_7']
    x_list.append(a)
# Create empty DataFrames in the forecast dict
forecast_dict['expert_2021']=pd.DataFrame()
forecast_dict['expert_2022']=pd.DataFrame()
# Create to empty dicts to put the forecasts results 
exp21={'Price_REG1':[],'Price_REG2':[],'Price_REG3':[],'Price_REG4':[]}
exp22={'Price_REG1':[],'Price_REG2':[],'Price_REG3':[],'Price_REG4':[]}

    
############################## Building block #################################
###############################################################################
###############################################################################
# 17544 is for 2021, 26304 for 2022
year_list = [17544,26304]
for year in year_list:
    # Number of times the calibration window is moved (length of test set)
    for p in range(365):
        predict_1 = []
        predict_2 = []
        predict_3 = []
        predict_4 = []
        dff = []
        # Transforms the data separetly for each column, since it takes p as
        # parameter the data is re-transformed for every iteration. Using the same
        # data as is used in the regression.
        for column in df:
            transform(df[[column]], p, year)
            dff.append(c)
            
        # Create lagged values for 1, 2, and 7 days.
        for i, j in zip(range(4), range(1, 5)):
            for s in [24, 48, 168]:
                dff[i][f'Price_{s}_{j}'] = dff[i][f'Price_REG{j}'].shift(s)
    
        # Create two new columns which contain the min and max values from the previous day
        # Min value
        for i, j in zip(range(4), range(1, 5)):
            dff[i][f'Price_{j}_min'] = dff[i][f'Price_REG{j}'].between_time(
                '00:00', '23:00').resample('d').min()
            dff[i][f'Price_{j}_min'].ffill(axis=0, inplace=True)
            dff[i][f'Price_{j}_min'] = dff[i][f'Price_{j}_min'].shift(24)
        # Max value
        for i, j in zip(range(4), range(1, 5)):
            dff[i][f'Price_{j}_max'] = dff[i][f'Price_REG{j}'].between_time(
                '00:00', '23:00').resample('d').max()
            dff[i][f'Price_{j}_max'].ffill(axis=0, inplace=True)
            dff[i][f'Price_{j}_max'] = dff[i][f'Price_{j}_max'].shift(24)
    
        # Create new column which is the last value from the previous day
        # A copy of the price by shifted one spot downwards
        for i, j in zip(range(4), range(1, 5)):
            dff[i][f'Price_last_{j}'] = dff[i][f'Price_REG{j}'].shift(1)
        # Fill up the other 23 values with the same values as created above
        # 1462 is the number of days between 20/01/01 - 22/12/31
        for i, j in zip(range(1, 1462), range(2, 1462)):
            for s, m in zip(range(4), range(1, 5)):
                dff[s][f'Price_last_{m}'].iloc[24*i:24*j]= dff[s][f'Price_REG{m}'].iloc[24*i-1]
        # Create daily dummies
        for i in range(4):
            dff[i].reset_index(inplace=True)
            dff[i]['Day of Week'] = dff[i]['date'].dt.dayofweek + 1
            dff[i][day_list] = pd.get_dummies(dff[i]['Day of Week'], prefix='day')
            dff[i].drop(['Day of Week'], axis=1, inplace=True)
            dff[i].set_index('date', inplace=True)
                   
############################### OLS Block #####################################
        # For loop to estimate model. For every pth iteration the estimation window
        # moves one step forward.
        for i, s, m, w, column in zip(range(4), range(1, 5), predict_list, x_list, df):
            model = sm.OLS(dff[i][f'Price_REG{s}'].iloc[(168+8760)+(p*24):(year)+(p*24)],
                       dff[i][w].iloc[(168+8760)+(p*24):(year)+(p*24)]).fit()
            # Multiply coefficients with each value for all 24 hours per day
            for j in range(24):
                predict = sum(model.params*dff[i][w].iloc[year+(p*24)+j])
                # run the transformation function and then inverse the transformed
                # values using the inverse mad-asinh
                transform(df[[column]], p, year)
                inverse(predict)
                # Put in correct place based on year and price zone
                if year == 17544 and s == 1:
                    exp21['Price_REG1'].append(c)
                elif year == 17544 and s == 2:
                    exp21['Price_REG2'].append(c)
                elif year == 17544 and s == 3:
                    exp21['Price_REG3'].append(c)
                elif year == 17544 and s == 4:
                    exp21['Price_REG4'].append(c)
                elif year == 26304 and s == 1:
                    exp22['Price_REG1'].append(c)
                elif year == 26304 and s == 2:
                    exp22['Price_REG2'].append(c)
                elif year == 26304 and s == 3:
                    exp22['Price_REG3'].append(c)
                elif year == 26304 and s == 4:
                    exp22['Price_REG4'].append(c)

###############################################################################
# Put forecasts into dict
for i in exp21:
    forecast_dict['expert_2021'][i]=exp21[i]
    forecast_dict['expert_2022'][i]=exp22[i]
    
###############################################################################
# Save as files
forecast_dict['expert_2021'].to_pickle(f"{path}\expert_forecast_21_short") 
forecast_dict['expert_2022'].to_pickle(f"{path}\expert_forecast_22_short") 


###############################################################################
del_list = ['a', 'b', 'c', 'column', 'day_list', 'dff', 'exp21', 'exp22',
            'inverse', 'j', 'm', 'median_abs_deviation', 
            'model', 'p', 'predict', 'predict_1', 'predict_2', 
            'predict_3', 'predict_4', 'predict_list', 's', 'sm', 
            'transform', 'w', 'x_list', 'year', 'year_list']
for i in del_list:
    del locals()[i]
del del_list
del i
###############################################################################
###############################################################################
###############################################################################












# Can be used to compute errors. Function to compute errors from all models
# are in the "EPF_result" file
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# Compute errors
expert_list = (['expert_2022', 'expert_2021'])
location_list_1 = ([26304, 17544])
location_list_2 = ([35064, 26304])

error_expert = np.zeros((2,8))
for i, j, s, m in zip(range(2), expert_list, location_list_1, location_list_2):
    for column, p in zip(df, range(4)):
        rmse = mean_squared_error(df[column].iloc[s:m], forecast_dict[j][column], squared=False)
        mae = mean_absolute_error(df[column].iloc[s:m], forecast_dict[j][column])
        error_expert[i,2*p:2*p+2] = [mae, rmse]
        
########################## LATEX ##############################################
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Create DF From Array
error_expert = pd.DataFrame(error_expert, columns = ['RMSE','MAE','RMSE','MAE','RMSE','MAE','RMSE','MAE'], index = ['AR Expert', 'AR Model'])
# Print LATEX Table
print(error_expert.to_latex(index=True))  
###############################################################################



    

    





    






    








