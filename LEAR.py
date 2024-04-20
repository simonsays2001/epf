# -*- coding: utf-8 -*-
'''
Created on Wed Nov 23 20:43:32 2022

@author: simon

This file is running the LEAR models. It is based on the code developed by Lagos et. al.
and the instruction for tis library is found at:
    https://epftoolbox.readthedocs.io/en/latest/modules/started.html
Before this code can be used, the Git has to be clown inside the command prompt
(mini-conda for example) this is done by:
    
    git clone https://github.com/jeslago/epftoolbox.git
    cd epftoolbox
    
and is installed by:
    
    pip install .    
    
    To just put the pre-run forecasts into the "forecast_dict" the first two
    lines can be uncommented and executed.
'''

#forecast_dict['forecast_lear_2021']  = pd.read_pickle(f"{path}\lear_2021_short")
#forecast_dict['forecast_lear_2022']  = pd.read_pickle(f"{path}\lear_2022_short")



import time
import os
from epftoolbox.models import evaluate_lear_in_test_dataset
import pandas as pd

    
path = r"C:\Users\simon\OneDrive\Documents\Skola\HT22\EPF_project\data"
df = pd.read_pickle(f"{path}\df")

df.set_index('date', inplace=True)
###############################################################################

# Create DataFrames in right format for the LEAR model
for i in range(1,5):
    a = df[[f'Price_REG{i}', f'SE{i}_C', f'SE{i}_W']]
    a = a.reset_index()
    a.rename(columns = {'date':'Date', f'Price_REG{i}':'Price', f'SE{i}_C':'Grid load forecast', f'SE{i}_W':'Wind power forecast'}, inplace = True)
    a.set_index('Date', inplace = True)
    a.to_csv(f"{path}/datasets/NP_{i}.csv")
# (Here a new folder "datasets" might have to be created in the path destination)


############################ LEAR for zone 1 2022 #############################

start_time = time.time()
# For zone 2 this is changed to NP_2 and so on.
dataset = 'NP_1'

# Number of days used in the training dataset for recalibration (364 per year (52*7))
calibration_window = 364*2
# Change according to test year
begin_test_date = "01/01/2021 00:00"
end_test_date = "31/12/2021 23:00"

path_datasets_folder = os.path.join(f"{path}\datasets")
# Change to 2022 if testing for that year
path_recalibration_folder = os.path.join(f"{path}\experimental_files_2021_2023_02_23")
    
evaluate_lear_in_test_dataset(path_recalibration_folder=path_recalibration_folder, 
                             path_datasets_folder=path_datasets_folder, dataset=dataset, 
                             calibration_window=calibration_window, begin_test_date=begin_test_date, 
                             end_test_date=end_test_date)
print("--- %s seconds ---" % (time.time() - start_time))

####################################################0##########################

############################ LEAR for zone 2 2022 #############################

start_time = time.time()
# For zone 2 this is changed to NP_2 and so on.
dataset = 'NP_2'

# Number of days used in the training dataset for recalibration (364 per year (52*7))
calibration_window = 364*2
# Change according to test year
begin_test_date = "01/01/2021 00:00"
end_test_date = "31/12/2021 23:00"

path_datasets_folder = os.path.join(f"{path}\datasets")
# Change to 2022 if testing for that year
path_recalibration_folder = os.path.join(f"{path}\experimental_files_2021_2023_02_23")
    
evaluate_lear_in_test_dataset(path_recalibration_folder=path_recalibration_folder, 
                             path_datasets_folder=path_datasets_folder, dataset=dataset, 
                             calibration_window=calibration_window, begin_test_date=begin_test_date, 
                             end_test_date=end_test_date)
print("--- %s seconds ---" % (time.time() - start_time))

####################################################0##########################

############################ LEAR for zone 3 2022 #############################

start_time = time.time()
# For zone 2 this is changed to NP_2 and so on.
dataset = 'NP_3'

# Number of days used in the training dataset for recalibration (364 per year (52*7))
calibration_window = 364*2
# Change according to test year
begin_test_date = "01/01/2021 00:00"
end_test_date = "31/12/2021 23:00"

path_datasets_folder = os.path.join(f"{path}\datasets")
# Change to 2022 if testing for that year
path_recalibration_folder = os.path.join(f"{path}\experimental_files_2021_2023_02_23")
    
evaluate_lear_in_test_dataset(path_recalibration_folder=path_recalibration_folder, 
                             path_datasets_folder=path_datasets_folder, dataset=dataset, 
                             calibration_window=calibration_window, begin_test_date=begin_test_date, 
                             end_test_date=end_test_date)
print("--- %s seconds ---" % (time.time() - start_time))

####################################################0##########################

############################ LEAR for zone 4 2021 #############################

start_time = time.time()
# For zone 2 this is changed to NP_2 and so on.
dataset = 'NP_4'

# Number of days used in the training dataset for recalibration (364 per year (52*7))
calibration_window = 364*2
# Change according to test year
begin_test_date = "01/01/2021 00:00"
end_test_date = "31/12/2021 23:00"

path_datasets_folder = os.path.join(f"{path}\datasets")
# Change to 2022 if testing for that year
path_recalibration_folder = os.path.join(f"{path}\experimental_files_2021_2023_02_23")
    
evaluate_lear_in_test_dataset(path_recalibration_folder=path_recalibration_folder, 
                             path_datasets_folder=path_datasets_folder, dataset=dataset, 
                             calibration_window=calibration_window, begin_test_date=begin_test_date, 
                             end_test_date=end_test_date)
print("--- %s seconds ---" % (time.time() - start_time))

###############################################################################
###############################################################################
###############################################################################


# Load the forecasts from csv file and store in list
forecast_lear_21 = []
for i in range(1,5):
    forecast_lear_i = pd.read_csv(f"{path}\experimental_files_2021_short\LEAR_forecast_datNP_{i}_YT2_CW728.csv", index_col=0).stack().to_frame()
    forecast_lear_i.reset_index(inplace=True)
    forecast_lear_i.drop(['Date', 'level_1'], inplace=True, axis=1)
    forecast_lear_i.rename(columns={0: f'Price_REG{i}'}, inplace=True)
    forecast_lear_21.append(forecast_lear_i)

forecast_lear_22 = []
for i in range(1,5):
    forecast_lear_i = pd.read_csv(f"{path}\experimental_files_2022_2023_02_23\LEAR_forecast_datNP_{i}_YT2_CW728.csv", index_col=0).stack().to_frame()
    forecast_lear_i.reset_index(inplace=True)
    forecast_lear_i.drop(['Date', 'level_1'], inplace=True, axis=1)
    forecast_lear_i.rename(columns={0: f'Price_REG{i}'}, inplace=True)
    forecast_lear_22.append(forecast_lear_i)

# Put dataframes in dictionary
forecast_dict['forecast_lear_2021']=pd.concat(forecast_lear_21, axis=1)
forecast_dict['forecast_lear_2022']=pd.concat(forecast_lear_22, axis=1)

# Save dataframes as a file
#forecast_dict['forecast_lear_2021'].to_pickle(f"{path}\lear_2021_short") 
#forecast_dict['forecast_lear_2022'].to_pickle(f"{path}\lear_2022_short") 

###############################################################################
del_list = ['forecast_lear_21', 'forecast_lear_22', 'forecast_lear_i', 
            'path_datasets_folder', 'path_recalibration_folder', 'start_time',
            'time', 'end_test_date', 'begin_test_date', 'dataset', 'calibration_window', 
            'a']
for i in del_list:
    del locals()[i]
del i 
del del_list
###############################################################################



















    
    
    
    
    
    
    



