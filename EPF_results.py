# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 08:40:23 2022

@author: simon
"""
pd.set_option('display.float_format', lambda x: '%.2f' % x)

'''
This file is used to create the figures and tables. It is divided by the chapter which
where they appear in the thesis. The theory section prints a timeline of the NordPool
routines. The Data section prints graphs and tables of the DA, wind, and consumption
data. The results section prints the table containing the MAE and RMSE and the
multivariate DM-graphs + the univariate DM-tables.
'''

# Path where the excel files, and provided DataFrames are placed.
path = r"C:\Users\simon\OneDrive\Documents\Skola\HT22\EPF_project\data"
# Path where figures are saved
figure_path = r"C:\Users\simon\OneDrive\Documents\Skola\HT22\EPF_project\figures"
# DataFrames containing consumption + wind + DA prices
df = pd.read_pickle(f"{path}\df")
# Same as above but also containing dummies for hour, day and week
df_dummies = pd.read_pickle(f"{path}\df_dummies")



###### Descriptive Statistics ######

## Time Line ##



import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np


############################ THEORY ####################################
########################################################################
########################################################################

# Nordpool timeline illustration

dates = [datetime(2021, 12, 30, 12, 45), datetime(2021, 12, 31, 0, 0), datetime(2021, 12, 31, 10, 0),
         datetime(2021, 12, 31, 12, 45), datetime(2022, 1, 1, 0, 0)]
min_date = datetime(np.min(dates).year, np.min(dates).month, np.min(dates).day, np.min(dates).hour - 6)
max_date = datetime(np.max(dates).year, np.max(dates).month, np.max(dates).day, np.max(dates).hour + 6)
 
labels = ['Hourly clearing prices for 12-31\nare announced to the market', '', 'Available capacities for 01-01\nare published', 
          'Hourly clearing prices for 01-01\nare announced to the market', '', '']
# labels with associated dates
labels = ['{0:%Y-%m-%d %H:%M}\n{1}'.format(d, l) for l, d in zip (labels, dates)]

fig, ax = plt.subplots(figsize=(15, 4), constrained_layout=True)
_ = ax.set_ylim(-2, 1.75)
_ = ax.set_xlim(min_date, max_date)
_ = ax.axhline(0, xmin=0.05, xmax=0.95, c='deeppink', zorder=1)
 
_ = ax.scatter(dates, np.zeros(len(dates)), s=120, c='palevioletred', zorder=2)
_ = ax.scatter(dates, np.zeros(len(dates)), s=30, c='darkmagenta', zorder=3)



label_offsets = np.zeros(len(dates))
label_offsets[0] = 0.35
label_offsets[2] = 0.35
label_offsets[4] = 0.165
#label_offsets[5] = 0.165
label_offsets[1] = -0.75
label_offsets[3] = -1
for i, (l, d) in enumerate(zip(labels, dates)):
    _ = ax.text(d, label_offsets[i], l, ha='center', fontfamily='serif', fontweight='bold', color='royalblue',fontsize=16)
    

stems = np.zeros(len(dates))
stems[::] = 0.3
stems[1] = -0.3
stems[3] = -0.3 
markerline, stemline, baseline = ax.stem(dates, stems, use_line_collection=True)
_ = plt.setp(markerline, marker=',', color='darkmagenta')
_ = plt.setp(stemline, color='darkmagenta')


# Hide lines around chart
for spine in ["left", "top", "right", "bottom"]:
    _ = ax.spines[spine].set_visible(False)
 
# Hide tick labels
_ = ax.set_xticks([])
_ = ax.set_yticks([])

plt.savefig(f"{figure_path}/time_line.pdf")
plt.show()

del_list = ['ax', 'dates', 'fig', 'markerline', 'max_date', 'min_date', 'spine', 
            'stemline', 'stems', 'label_offsets', 'l', 'i', 'labels', 'baseline', 'd' ]
for i in del_list:
    del locals()[i]



################################# SECTION 3 -  DATA #######################################
###########################################################################################
###########################################################################################


describe = df[['Price_REG1', 'Price_REG2', 'Price_REG3', 'Price_REG4']].describe()
describe = describe.T
describe.drop(['25%', '50%', '75%'], axis=1, inplace=True)
skew = df[['Price_REG1', 'Price_REG2', 'Price_REG3', 'Price_REG4']].skew()
kurt = df[['Price_REG1', 'Price_REG2', 'Price_REG3', 'Price_REG4']].kurtosis()
describe['skewness'] = skew.tolist()
describe['kurtosis'] = kurt.tolist()
print(describe.to_latex(index=True))  

describe = df[['SE1_W', 'SE2_W', 'SE3_W', 'SE4_W']].describe()
describe = describe.T
describe.drop(['25%', '50%', '75%'], axis=1, inplace=True)
skew = df[['SE1_W', 'SE2_W', 'SE3_W', 'SE4_W']].skew()
kurt = df[['SE1_W', 'SE2_W', 'SE3_W', 'SE4_W']].kurtosis()
describe['skewness'] = skew.tolist()
describe['kurtosis'] = kurt.tolist()
print(describe.to_latex(index=True))  

describe = df[['SE1_C', 'SE2_C', 'SE3_C', 'SE4_C']].describe()
describe = describe.T
describe.drop(['25%', '50%', '75%'], axis=1, inplace=True)
skew = df[['SE1_C', 'SE2_C', 'SE3_C', 'SE4_C']].skew()
kurt = df[['SE1_C', 'SE2_C', 'SE3_C', 'SE4_C']].kurtosis()
describe['skewness'] = skew.tolist()
describe['kurtosis'] = kurt.tolist()
print(describe.to_latex(index=True))

del skew
del kurt
del describe

######################################################################

# Time Series of data

# Day-Ahead
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(8,8))
ax1.set_title('Zone 1')
ax1.plot(df['Price_REG1'], lw=0.5)
ax2.set_title('Zone 2')
ax2.plot(df['Price_REG2'], lw=0.5)
ax3.set_title('Zone 3')
ax3.plot(df['Price_REG3'], lw=0.5)
ax4.set_title('Zone 4')
ax4.plot(df['Price_REG4'], lw=0.5)
fig.tight_layout()
plt.savefig(f"{figure_path}/time_series_4_1_short.pdf")
plt.show()

df.set_index('date', inplace=True)

# Wind prognosis
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(8,8))
ax1.set_title('Zone 1')
ax1.plot(df['SE1_W'], lw=0.5)
ax2.set_title('Zone 2')
ax2.plot(df['SE2_W'], lw=0.5)
ax3.set_title('Zone 3')
ax3.plot(df['SE3_W'], lw=0.5)
ax4.set_title('Zone 4')
ax4.plot(df['SE4_W'], lw=0.5)
fig.tight_layout()
plt.savefig(f"{figure_path}/time_series_wind_short.pdf")
plt.show()

# Consumption
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(8,8))
ax1.set_title('Zone 1')
ax1.plot(df['SE1_C'], lw=0.5)
ax2.set_title('Zone 2')
ax2.plot(df['SE2_C'], lw=0.5)
ax3.set_title('Zone 3')
ax3.plot(df['SE3_C'], lw=0.5)
ax4.set_title('Zone 4')
ax4.plot(df['SE4_C'], lw=0.5)
fig.tight_layout()
plt.savefig(f"{figure_path}/time_series_consumption_short.pdf")
plt.show()

################################################################################

# Histogram Day-Ahead
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=False, figsize=(7,5))
ax1.set_title('Zone 1')
ax1.hist(df['Price_REG1'], lw=0.5, bins=100)
ax2.set_title('Zone 2')
ax2.hist(df['Price_REG2'], lw=0.5, bins=100)
ax3.set_title('Zone 3')
ax3.hist(df['Price_REG3'], lw=0.5, bins=100)
ax4.set_title('Zone 4')
ax4.hist(df['Price_REG4'], lw=0.5, bins=100)
fig.tight_layout()
plt.savefig(f"{figure_path}/hist_2_2.pdf")
plt.show()


df_mean = df_dummies[['Price_REG1', 'Price_REG2', 'Price_REG3', 'Price_REG4', 'Hour of Week']]
########################## Some averages #######################################
# Create mean, first quartile and third qurtile of the data.
df_168h_mean = df_mean.groupby('Hour of Week').mean(numeric_only=True)
df_168h_25 = df_mean.groupby('Hour of Week').quantile(0.25, numeric_only=True)
df_168h_75 = df_mean.groupby('Hour of Week').quantile(0.75, numeric_only=True)
###############################################################################
# Remove suoerflous columns
df_168h_mean = df_168h_mean[['Price_REG1', 'Price_REG2', 'Price_REG3', 'Price_REG4']]
df_168h_25 = df_168h_25[['Price_REG1', 'Price_REG2', 'Price_REG3', 'Price_REG4']]
df_168h_75 = df_168h_75[['Price_REG1', 'Price_REG2', 'Price_REG3', 'Price_REG4']]
###############################################################################

# Weekly averages 
fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=False, figsize=(10,5))
plt.xticks([0,24,48,72,96,120,144,168])
ax1.plot(df_168h_mean.Price_REG1, label='Zone 1')
ax1.plot(df_168h_mean.Price_REG2, label='Zone 2')
ax1.plot(df_168h_mean.Price_REG3, label='Zone 2')
ax1.plot(df_168h_mean.Price_REG4, label='Zone 4')
ax1.legend(loc="upper right", bbox_to_anchor=[1, 1],
                 ncol=2, shadow=True, fancybox=True)
ax1.axhline(df_168h_mean.iloc[0:23].mean(axis=1).mean(), xmin=0.072, xmax=0.182, color='blue', ls='dotted')
ax1.axhline(df_168h_mean.iloc[24:47].mean(axis=1).mean(), xmin=0.202, xmax=0.312, color='blue', ls='dotted')
ax1.axhline(df_168h_mean.iloc[48:71].mean(axis=1).mean(), xmin=0.333, xmax=0.443, color='blue', ls='dotted')
ax1.axhline(df_168h_mean.iloc[72:95].mean(axis=1).mean(), xmin=0.463, xmax=0.573, color='blue', ls='dotted')
ax1.axhline(df_168h_mean.iloc[96:119].mean(axis=1).mean(), xmin=0.595, xmax=0.705, color='blue', ls='dotted')
ax1.axhline(df_168h_mean.iloc[120:143].mean(axis=1).mean(), xmin=0.73, xmax=0.85, color='blue', ls='dotted')
ax1.axhline(df_168h_mean.iloc[144:167].mean(axis=1).mean(), xmin=0.85, xmax=0.97, color='blue', ls='dotted')
fig.tight_layout()
plt.savefig(f"{figure_path}/weekly_averages_short.pdf")
plt.show()

# 4X1 weekly averages with confidence interval 25, 75.
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=False, figsize=(10,8))
plt.xticks([0,24,48,72,96,120,144,168])
ax1.plot(df_168h_mean.Price_REG1)
ax1.fill_between(range(168), df_168h_25.Price_REG1, df_168h_75.Price_REG1,alpha=.35)
ax1.set_title('Zone 1')
ax2.plot(df_168h_mean.Price_REG2)
ax2.fill_between(range(168), df_168h_25.Price_REG2, df_168h_75.Price_REG2,alpha=.35)
ax2.set_title('Zone 2')
ax3.plot(df_168h_mean.Price_REG3)
ax3.fill_between(range(168), df_168h_25.Price_REG3, df_168h_75.Price_REG3,alpha=.35)
ax3.set_title('Zone 3')
ax4.plot(df_168h_mean.Price_REG4)
ax4.fill_between(range(168), df_168h_25.Price_REG4, df_168h_75.Price_REG4,alpha=.35)
ax4.set_title('Zone 4')
fig.tight_layout()
plt.savefig(f"{figure_path}/weekly_averages_4_1_short.pdf")
plt.show()


############################# RESULT ##############################################
###################################################################################
###################################################################################


# Results table. Only two naive forecasts are included.

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


df = df[['Price_REG1', 'Price_REG2', 'Price_REG3', 'Price_REG4']]

error_list = (['forecast_24_', 'forecast_standard_', 'expert_', 'forecast_lear_'])
location_list_1 = ([17544, 26304])
location_list_2 = ([26304, 35064])
year_list = ['2021', '2022']
error = np.zeros((8,8))
for i, j in zip(range(0,7,2), error_list):
    for s, m, n, year in zip(location_list_1, location_list_2, range(2), year_list):
        for column, p in zip(df, range(4)):
            rmse = mean_squared_error(df_dummies[column].iloc[s:m], forecast_dict[j+year][column], squared=False)
            mae = mean_absolute_error(df_dummies[column].iloc[s:m], forecast_dict[j+year][column])
            error[i+n,2*p:2*p+2] = [mae, rmse]
# Create Dataframe from errors
error = pd.DataFrame(error, columns = ['MAE','RMSE','MAE','RMSE','MAE','RMSE','MAE', 'RMSE'],
                          index = ['24_21', '24_22', 'st_21',
                                   'st22', 'AR_21', 'AR_22', 'LEAR_21',
                                   'LEAR_22'])
# Print LATEX Table
print(error.to_latex(index=True))  

###############################################################################
################################## DM TESTS ###################################
###############################################################################

# Create dataframes from the DM tests. Total of 8. Each containing all the 
# forecasts from each year and each price zone. 2 years * 4 price ones = 8. 
g = ['21_1', '21_2', '21_3', '21_4', '22_1', '22_2', '22_3', '22_4']
year_list = [2021, 2022]
year_list2 = ['predictions_21', 'predictions_22']
zone_list = ['_1', '_2', '_3', '_4']
prediction_dict = {}
for p, q in zip(year_list, year_list2):
    for j, s, d in zip(range(1,5), range(4), zone_list):
        a = pd.concat([forecast_dict[f'forecast_24_{p}'][f'Price_REG{j}'], 
                       forecast_dict[f'forecast_standard_{p}'][f'Price_REG{j}'], 
                       forecast_dict[f'expert_{p}'][f'Price_REG{j}'], forecast_dict[f'forecast_lear_{p}'][f'Price_REG{j}']], axis=1)
        prediction_dict[q + d]=a
        prediction_dict[q + d].columns = ['Naive 1', 'Naive 3', 'AR', 'LEAR',]
        
        
# To drop the 'Auto' column
for i in year_list2:
    for j in zone_list:
        prediction_dict[f'{i}{j}'].drop(['Auto'], axis=1,inplace=True)
        
###############################################################################
del_list = ['g', 'i', 'j', 'p', 'q', 's', 
            'a', 'd', 'i', 'year_list', 'year_list2', 'zone_list']
for i in del_list:
    del locals()[i]
del i 
del del_list
###############################################################################




###############################################################################
######################## MULTIVARIATE DM TEST #################################
###############################################################################

from epftoolbox.evaluation import DM, plot_multivariate_DM_test
from epftoolbox.data import read_data
import matplotlib.pyplot as plt

###############################################################################

# 2021
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(8,8))
plt.sca(ax1)
plot_multivariate_DM_test(df['Price_REG1'].iloc[17544:26304],
                          prediction_dict['predictions_21_1'], norm=1, title='Zone 1', savefig=False, path='')
plt.sca(ax2)
plot_multivariate_DM_test(df['Price_REG2'].iloc[17544:26304],
                          prediction_dict['predictions_21_2'], norm=1, title='Zone 2', savefig=False, path='')
plt.sca(ax3)
plot_multivariate_DM_test(df['Price_REG3'].iloc[17544:26304],
                          prediction_dict['predictions_21_3'], norm=1, title='Zone 3', savefig=False, path='')
plt.sca(ax4)
plot_multivariate_DM_test(df['Price_REG4'].iloc[17544:26304],
                          prediction_dict['predictions_21_4'], norm=1, title='Zone 4', savefig=False, path='')
fig.tight_layout()
fig.savefig(f"{figure_path}\DM_2021_SHORT.pdf")
plt.show()

###############################################################################

# 2022
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(8,8))
plt.sca(ax1)
plot_multivariate_DM_test(df['Price_REG1'].iloc[26304:35064],
                         prediction_dict['predictions_22_1'], norm=1, title='Zone 1', savefig=False, path='')
plt.sca(ax2)
plot_multivariate_DM_test(df['Price_REG2'].iloc[26304:35064],
                          prediction_dict['predictions_22_2'], norm=1, title='Zone 2', savefig=False, path='')
plt.sca(ax3)
plot_multivariate_DM_test(df['Price_REG3'].iloc[26304:35064],
                          prediction_dict['predictions_22_3'], norm=1, title='Zone 3', savefig=False, path='')
plt.sca(ax4)
plot_multivariate_DM_test(df['Price_REG4'].iloc[26304:35064],
                          prediction_dict['predictions_22_4'], norm=1, title='Zone 4', savefig=False, path='')
fig.tight_layout()
fig.savefig(f"{figure_path}\DM_2022_SHORT.pdf")
plt.show()

###############################################################################
del_list = ['ax1', 'ax2', 'ax3', 'ax4', 'fig']
for i in del_list:
    del locals()[i]
###############################################################################



###############################################################################
########################### UNIVARIATE DM TEST ################################
###############################################################################

# Extract 2021 and 2022 "real" prices
df_list = ['df_21_1', 'df_21_2',  'df_21_3', 'df_21_4',
           'df_22_1', 'df_22_2', 'df_22_3', 'df_22_4']
start_list = [17544, 26304, 17544, 26304,
              17544, 26304, 17544, 26304]
end_list = [26304, 35064, 26304, 35064, 
            26304, 35064, 26304, 35064]
range_list = [1,2,3,4,1,2,3,4]

# Place them in dict
df_dict = {key: df[f'Price_REG{i}'].iloc[s:m] for key, i,
           s, m in zip(df_list, range_list, start_list, end_list)}

# Make indexes start from 0
for i in range(21,23):
    for j in range(1,5):
        df_dict[f"df_{i}_{j}"] = pd.DataFrame(df_dict[f"df_{i}_{j}"])
        df_dict[f"df_{i}_{j}"].reset_index(inplace=True)
        df_dict[f"df_{i}_{j}"].drop(columns={'date'}, inplace=True)
        
###############################################################################


# Univariate test statistics (without 'Auto')
pd.options.display.float_format = '{:.10f}'.format 
np.set_printoptions(precision=20, suppress=True)

# Create univaraite test statistics for all models and both years
dm1 = {}
dm2 = {}
lst = ['_21', '_22']
dm_list = ['Naive_expert_', 'Naive_LEAR_', 'LEAR_Expert_']
first_model = ['Naive 1', 'Naive 1', 'LEAR']
second_model = ['AR', 'LEAR', 'AR']
zone_list = ['1','2','3','4']
for s, q, h in zip(dm_list, first_model, second_model):
    for i in lst:
        for j, p in zip(range(1,5), zone_list):
            a = DM(p_real=df_dict[f"df{i}_{j}"].values.reshape(-1, 24),
               p_pred_1=prediction_dict[f"predictions{i}_{j}"][q].values.reshape(-1, 24),
               p_pred_2=prediction_dict[f"predictions{i}_{j}"][h].values.reshape(-1, 24), 
               norm=1, version='univariate')
            if i =='_21':
                dm1[s + p + i]=a
            else:
                dm2[s + p + i]=a



'''
# Create univaraite test statistics for all models and both years
dm1 = {}
dm2 = {}
lst = ['_21', '_22']
dm_list = ['Naive_expert_', 'Naive_LEAR_', 'Naive_Auto_', 'LEAR_Expert_',
           'LEAR_Auto_', 'Expert_Auto_' ]
first_model = ['Naive 1', 'Naive 1', 'Naive 1', 'LEAR', 'LEAR', 'AR']
second_model = ['AR', 'LEAR', 'Auto', 'AR', 'Auto', 'Auto']
zone_list = ['1','2','3','4']
for s, q, h in zip(dm_list, first_model, second_model):
    for i in lst:
        for j, p in zip(range(1,5), zone_list):
            a = DM(p_real=df_dict[f"df{i}_{j}"].values.reshape(-1, 24),
               p_pred_1=prediction_dict[f"predictions{i}_{j}"][q].values.reshape(-1, 24),
               p_pred_2=prediction_dict[f"predictions{i}_{j}"][h].values.reshape(-1, 24), 
               norm=1, version='univariate')
            if i =='_21':
                dm1[s + p + i]=a
            else:
                dm2[s + p + i]=a
 '''          
        
###############################################################################

# Make combined DataFrame from all test statistics   
arr_lst21 = ['Naive_expert_1_21', 'Naive_LEAR_1_21', 'LEAR_Expert_1_21', 'Naive_expert_2_21', 'Naive_LEAR_2_21',
  'LEAR_Expert_2_21', 'Naive_expert_3_21', 'Naive_LEAR_3_21', 'LEAR_Expert_3_21', 
  'Naive_expert_4_21', 'Naive_LEAR_4_21', 'LEAR_Expert_4_21']
arr_lst22 = ['Naive_expert_1_22', 'Naive_LEAR_1_22', 'LEAR_Expert_1_22', 'Naive_expert_2_22', 'Naive_LEAR_2_22',
  'LEAR_Expert_2_22', 'Naive_expert_3_22', 'Naive_LEAR_3_22', 'LEAR_Expert_3_22', 
  'Naive_expert_4_22', 'Naive_LEAR_4_22', 'LEAR_Expert_4_22'] 
   
df_from_arr_21 = pd.DataFrame(data=[dm1[x] for x in arr_lst21])
df_from_arr_22 = pd.DataFrame(data=[dm2[x] for x in arr_lst22])
# Transpose
df_from_arr_21 = df_from_arr_21.T
df_from_arr_22 = df_from_arr_22.T
# Make index start from 1
df_from_arr_21.index = np.arange(1, len(df_from_arr_21) + 1)
df_from_arr_22.index = np.arange(1, len(df_from_arr_22) + 1)
# Print LaTeX Tables
print(df_from_arr_21.to_latex(index=True))  
print(df_from_arr_22.to_latex(index=True))  

###############################################################################

# Not customized to run without 'Auto'

# Create table that prints the best model for each hour
# 2021
error = np.zeros((24,8), dtype='<U8')
for j, s in zip(range(0,10,3), range(4)):
    for i in range(24):
        if df_from_arr_21[1+j].iloc[i] < 0.5 and df_from_arr_21[2+j].iloc[i] > 0.5:
            error[i,s*2] = "LEAR"
        elif df_from_arr_21[0+j].iloc[i] > 0.5 and df_from_arr_21[1+j].iloc[i] > 0.5:
            error[i,s*2] = "Naive"
        elif df_from_arr_21[0+j].iloc[i] < 0.5 and df_from_arr_21[2+j].iloc[i] < 0.5:
            error[i,s*2] = "AR"
# 2022            
for j, s in zip(range(0,10,3), range(4)):
    for i in range(24):
        if df_from_arr_22[1+j].iloc[i] < 0.5 and df_from_arr_22[2+j].iloc[i] > 0.5:
            error[i,(s*2+1)] = "LEAR"
        elif df_from_arr_22[0+j].iloc[i] > 0.5 and df_from_arr_22[1+j].iloc[i] > 0.5:
            error[i,(s*2+1)] = "Naive"
        elif df_from_arr_22[0+j].iloc[i] < 0.5 and df_from_arr_22[2+j].iloc[i] < 0.5:
            error[i,(s*2+1)] = "AR"
                   
df_from_error = pd.DataFrame(data=error)
df_from_error.index = np.arange(1, 25)
print(df_from_error.to_latex(index=True))  

aa = np.zeros((3,8))
aa[0:2,0] = (df_from_error[0].value_counts())

df_from_error[0].value_counts(sort=False).reindex(df_from_error.unique(), fill_value=0)

aa=df_from_error[0].value_counts()
aa1=df_from_error[1].value_counts()
aa3 = np.zeros((2,2))
aa3= np.concatenate((aa,aa1),axis=1)
aa3=np.hstack((aa,aa1))
aa3.reshape(2,2)