
'''
- Creates the Naive forcasts. First four different forecasts are created, then
- the "forecast_dict" is created and the forecasts are stored there.
'''

import pandas as pd
import numpy as np
# Path where the excel files, and provided DataFrames are placed.
path = r"C:\Users\simon\OneDrive\Documents\Skola\HT22\EPF_project\data"
df_dummies = pd.read_pickle(f"{path}\df_dummies")




# Simple naive forecasts
forecast_24 = df_dummies[['Price_REG1', 'Price_REG2', 'Price_REG3','Price_REG4'
                      ]].shift(24)
forecast_168 = df_dummies[['Price_REG1', 'Price_REG2', 'Price_REG3','Price_REG4'
                      ]].shift(168)
forecast_24.reset_index(inplace=True)
forecast_168.reset_index(inplace=True)




# DF reset + add Day of Week Columns


########## STANDARD NAIVE ################
dow_167 = df_dummies[['Price_REG1', 'Price_REG2', 'Price_REG3','Price_REG4'
                      ]].loc[df_dummies['Day of Week'].isin([1,6,7])].shift(72)
dow_2345 = df_dummies[['Price_REG1', 'Price_REG2', 'Price_REG3', 'Price_REG4'
                       ]].loc[df_dummies['Day of Week'].isin([2,3,4,5])].shift(24)
dow_167.reset_index(inplace=True)
dow_2345.reset_index(inplace=True)
forecast_standard = pd.concat([dow_2345, dow_167], axis=0)
forecast_standard.sort_values(by='date', inplace = True)
forecast_standard.reset_index(inplace=True, drop=True)

########## CUSTOM NAIVE ################
dow_13567 = df_dummies[['Price_REG1', 'Price_REG2', 'Price_REG3','Price_REG4'
                      ]].loc[df_dummies['Day of Week'].isin([1,3,5,6,7])].shift(72)
dow_24 = df_dummies[['Price_REG1', 'Price_REG2', 'Price_REG3', 'Price_REG4'
                       ]].loc[df_dummies['Day of Week'].isin([2,4])].shift(24)
dow_13567.reset_index(inplace=True)
dow_24.reset_index(inplace=True)
forecast_custom = pd.concat([dow_24, dow_13567], axis=0)
forecast_custom.sort_values(by='date', inplace = True)
forecast_custom.reset_index(inplace=True, drop=True)


del_list = ['dow_13567', 'dow_167', 'dow_2345', 'dow_24']
for i in del_list:
    del locals()[i]

##############################################################################

forecast_24.to_pickle(f"{path}/forecast_24") 
forecast_168.to_pickle(f"{path}/forecast_168") 
forecast_custom.to_pickle(f"{path}/forecast_custom") 
forecast_standard.to_pickle(f"{path}/forecast_standard") 


# Divide into 2021 and 2022 and put into dict
forecast_list2 = ['forecast_24_2021', 'forecast_168_2021', 'forecast_standard_2021',
                  'forecast_custom_2021', 'forecast_24_2022', 'forecast_168_2022',
                  'forecast_standard_2022', 'forecast_custom_2022']
forecast_list = ['forecast_24', 'forecast_168', 'forecast_standard', 'forecast_custom']

forecast_dict = {}
start_date = [17544, 26304]
end_date = [26304, 35064]
year_list = ['2021','2022']
for s, m, q in zip(start_date, end_date, year_list):
    for j in forecast_list:
        a = locals()[j].iloc[s:m].copy()
        a.drop(['date'], inplace=True, axis=1)
        a.reset_index(inplace=True, drop=True)
        forecast_dict[j + '_' +q]=a
        
###############################################################################
del_list = ['a', 'end_date', 'forecast_list', 'forecast_list2', 'i', 'j', 'm',
            's', 'start_date', 'q', 'year_list', 'forecast_24', 
            'forecast_168', 'forecast_custom', 'forecast_standard']
for i in del_list:
    del locals()[i]
del i
del del_list
###############################################################################

















