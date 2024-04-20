# -*- coding: utf-8 -*-
'''
Created on Sun Jan 29 16:41:16 2023

@author: simon

Script to load the Auto models created in R. The first part loads the files and
transforms them. The second part can be used to create a LaTeX table with 
the error measures. 
'''


# Path where the excel files, and provided DataFrames are placed.
path = r"C:\Users\simon\OneDrive\Documents\Skola\HT22\EPF_project\data"

import pandas as pd
import numpy as np

price_list = [1, 2, 3, 4]
year_list = ['auto_2021', 'auto_2022']
year_2_list = [21, 22]

auto_2021 = pd.DataFrame()
auto_2022 = pd.DataFrame()

for i, m in zip(year_list, year_2_list):
    for j in price_list:
        locals()[i][j] = pd.read_csv(f"{path}/From R/auto_{j}_{m}.csv") 
for i in year_list:
    locals()[i].columns = ['Price_REG1', 'Price_REG2', 'Price_REG3',
                       'Price_REG4']
    forecast_dict[i]=locals()[i]

    
#############################################################################################  
del_list = ['i', 'j', 'm', 'price_list', 'year_2_list', 'year_list', 'auto_2021', 'auto_2022']
for i in del_list:
    del locals()[i]
del i 
del del_list
############################################################################################







