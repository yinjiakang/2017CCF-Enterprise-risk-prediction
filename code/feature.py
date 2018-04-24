
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import xgboost as xgb

from dateutil import parser
from datetime import datetime
import math
data_path = "../data/"
result_path =  "../data/result/"
feature_path =  "../data/feature/"
analysis_path =  "../data/analysis/"
function_path =  "../data/function/"

# reading data
train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'evaluation_public.csv')
entbase = pd.read_csv(data_path + '1entbase.csv')
alter = pd.read_csv(data_path + '2alter.csv')
branch = pd.read_csv(data_path + '3branch.csv')
invest = pd.read_csv(data_path + '4invest.csv')
right = pd.read_csv(data_path + '5right.csv')
project = pd.read_csv(data_path + '6project.csv')
lawsuit = pd.read_csv(data_path + '7lawsuit.csv')
breakfaith = pd.read_csv(data_path + '8breakfaith.csv')
recruit = pd.read_csv(data_path + '9recruit.csv')
qualification = pd.read_csv(data_path + '10qualification.csv')


# In[2]:

import math
def translateYear(date):
    if type(date) == float:
        return np.nan
    year = int(date[:4])
    month = int(date[-2:])
    return (year-2015)*12 + month

def translateYearALTER(date):
    if type(date) == float:
        return np.nan
    year = int(date[:4])
    month = int(date[-2:])
    return (year-2013)*12 + month

def translateYearRECRUIT(date):
    if type(date) == float:
        return np.nan
    year = int(date[:4])
    month = int(date[-2:])
    return (year-2008)*12 + month

def translateYearFB(date):
    if type(date) == float:
        return np.nan
    year = int(date[:4])
    month = int(date[5:-2])
    return (year-2010)*12 + month

import math
def translateEnddate(date):
    if type(date) == float:
        return 2015
    year = int(date[:4])
    return year

def get_HandE(new_right, col_name, prefix_index):
    right_type_HY_sta = new_right.groupby(['HY'])[col_name].agg({sum, max,min, np.mean}).add_prefix(col_name[: prefix_index] + '_HY_').reset_index()
    right_type_ETYPE_sta = new_right.groupby(['ETYPE'])[col_name].agg({sum, max,min, np.mean}).add_prefix(col_name[: prefix_index] + '_ETYPE_').reset_index()
    right_type_HYandETYPE_sta = new_right.groupby(['ETYPE', 'HY'])[col_name].agg({sum, max,min, np.mean}).add_prefix(col_name[: prefix_index] + '_HYandETYPE_').reset_index()
    
    new_right = pd.merge(new_right, right_type_HY_sta, on = 'HY', how = 'left')
    new_right = pd.merge(new_right, right_type_ETYPE_sta, on = 'ETYPE', how = 'left')
    new_right = pd.merge(new_right, right_type_HYandETYPE_sta, on = ['ETYPE', 'HY'], how = 'left') 
    
    new_right[col_name[:prefix_index] + '_HY_gap'] = new_right[col_name] - new_right[col_name[:prefix_index] + '_HY_mean'] 
    new_right[col_name[:prefix_index] + '_ETYPE_gap'] = new_right[col_name] - new_right[col_name[:prefix_index] + '_ETYPE_mean'] 
    new_right[col_name[:prefix_index] + '_HYandETYPE_gap'] = new_right[col_name] - new_right[col_name[:prefix_index] + '_HYandETYPE_mean']  
    
    return new_right

def get_HandEforEntbase(new_right, col_name, prefix_index):
    right_type_HY_sta = new_right.groupby(['HY'])[col_name].agg({max, np.mean, np.median}).add_prefix(col_name[: prefix_index] + '_HY_').reset_index()
    right_type_ETYPE_sta = new_right.groupby(['ETYPE'])[col_name].agg({ max, np.mean, np.median}).add_prefix(col_name[: prefix_index] + '_ETYPE_').reset_index()
    right_type_HYandETYPE_sta = new_right.groupby(['ETYPE', 'HY'])[col_name].agg({max, np.mean, np.median}).add_prefix(col_name[: prefix_index] + '_HYandETYPE_').reset_index()
    
    new_right = pd.merge(new_right, right_type_HY_sta, on = 'HY', how = 'left')
    new_right = pd.merge(new_right, right_type_ETYPE_sta, on = 'ETYPE', how = 'left')
    new_right = pd.merge(new_right, right_type_HYandETYPE_sta, on = ['ETYPE', 'HY'], how = 'left') 
    
    new_right[col_name[:prefix_index] + '_HY_gap'] = new_right[col_name] - new_right[col_name[:prefix_index] + '_HY_mean'] 
    new_right[col_name[:prefix_index] + '_ETYPE_gap'] = new_right[col_name] - new_right[col_name[:prefix_index] + '_ETYPE_mean'] 
    new_right[col_name[:prefix_index] + '_HYandETYPE_gap'] = new_right[col_name] - new_right[col_name[:prefix_index] + '_HYandETYPE_mean']  
    
    return new_right

def get_HandEandPforEntbase(df, col_name, prefix_index):
    df_PROV_sta = df.groupby(['PROV'])[col_name].agg({ max,np.mean, np.median}).add_prefix(col_name[: prefix_index]+ '_PROV_').reset_index()
    df_PROV_HY_sta = df.groupby(['PROV', 'HY'])[col_name].agg({ max,np.mean, np.median}).add_prefix(col_name[: prefix_index]+ '_PROVandHY_').reset_index()
    df_PROV_ETYPE_sta = df.groupby(['PROV', 'ETYPE'])[col_name].agg({max,  np.mean, np.median}).add_prefix(col_name[: prefix_index]+ '_PROVandETYPE_').reset_index()
    df_PROV_HY_ETYPE_sta = df.groupby(['PROV', 'HY', 'ETYPE'])[col_name].agg({max,np.mean, np.median}).add_prefix(col_name[: prefix_index]+ '_PROVandHYandETYPE_').reset_index()

    df = pd.merge(df, df_PROV_sta, on = 'PROV', how = 'left')
    df = pd.merge(df, df_PROV_HY_sta, on = ['PROV', 'HY'], how = 'left')
    df = pd.merge(df, df_PROV_ETYPE_sta, on = ['PROV', 'ETYPE'], how = 'left') 
    df = pd.merge(df, df_PROV_HY_ETYPE_sta, on = ['PROV', 'ETYPE', 'HY'], how = 'left') 
    
    
    df[col_name[:prefix_index] + '_PROV_gap'] = df[col_name] - df[col_name[:prefix_index] + '_PROV_mean'] 
    df[col_name[:prefix_index] + '_PROVandHY_gap'] = df[col_name] - df[col_name[:prefix_index] + '_PROVandHY_mean'] 
    df[col_name[:prefix_index] + '_PROVandETYPE_gap'] = df[col_name] - df[col_name[:prefix_index] + '_PROVandETYPE_mean']  
    df[col_name[:prefix_index] + '_PROVandHYandETYPE_gap'] = df[col_name] - df[col_name[:prefix_index] + '_PROVandHYandETYPE_mean']  

    return df    


# # Train

# In[3]:

train = pd.read_csv(data_path + 'train.csv')
train['ENDDATE_n'] = train.ENDDATE.apply(translateEnddate)
new_train = train.drop(['ENDDATE', 'ENDDATE_n'], axis = 1)



entbase = pd.read_csv(data_path + '1entbase.csv')

EID_HY_ETYPE = entbase[['EID', 'HY', 'ETYPE']]

entbase['EID_NUMBER'] = entbase.EID.apply(lambda row: row[1:])
entbase['EID_NUMBER'] = entbase.EID.apply(lambda row: row[1:])


entbase.ZCZB.fillna(0, inplace = True)
entbase.MPNUM.fillna(0, inplace = True)
entbase.INUM.fillna(0, inplace = True)
entbase.ENUM.fillna(0, inplace = True)
entbase.FINZB.fillna(0, inplace = True)
entbase.FSTINUM.fillna(0, inplace = True)
entbase.TZINUM.fillna(0, inplace = True)


new_entbase = entbase.copy()
new_entbase['RGYEAR_gap'] = 2016 - new_entbase.RGYEAR
new_entbase['ZCZB_mul_YEAR'] = new_entbase['RGYEAR_gap'] * 100 / 65 * new_entbase['ZCZB']



new_entbase['2014_Policy'] = new_entbase['RGYEAR'].apply(lambda row: 1 if row < 2014 else 2)


HY_null_index = list(new_entbase[new_entbase.HY.isnull()].index)


for index in HY_null_index:
    row = new_entbase.loc[index]
    etype = row.ETYPE
    etype_record = new_entbase[new_entbase.ETYPE == etype]
    fillHY = (etype_record.HY.value_counts().reset_index())['index'][0]
    row.HY = fillHY
    new_entbase.loc[index] = row


# In[573]:

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(2)
tmp = new_entbase.copy()
tmp.drop(['EID','EID_NUMBER', 'TZINUM'], axis = 1, inplace = True)
poly_feature = poly.fit_transform(tmp)
colname = [('Poly_f' + str(i)) for i in range(poly_feature.shape[1])]


# In[574]:

ploy_df = pd.DataFrame(poly_feature, columns = colname)
new_entbase = pd.concat([new_entbase, ploy_df], axis = 1)


# In[575]:

new_entbase['n_ZCZB_DEL_FINZB'] = new_entbase['ZCZB_mul_YEAR'] - new_entbase['FINZB']
new_entbase['n_ZCZB_ADD_FINZB'] = new_entbase['ZCZB_mul_YEAR'] + new_entbase['FINZB']
new_entbase['MP2-FSTI2'] = new_entbase['MPNUM']**2 - new_entbase['FSTINUM']**2

#1.10
new_entbase['MP2+FSTI2'] = new_entbase['MPNUM']**2 + new_entbase['FSTINUM']**2
new_entbase['I2+FSTI2'] = new_entbase['INUM']**2 + new_entbase['FSTINUM']**2
new_entbase['E2+FSTI2'] = new_entbase['ENUM']**2 + new_entbase['FSTINUM']**2
new_entbase['I2+MP2'] = new_entbase['INUM']**2 + new_entbase['MPNUM']**2
new_entbase['E2+MP2'] = new_entbase['ENUM']**2 + new_entbase['MPNUM']**2

new_entbase['RGYEAR_mul_FINZB'] = new_entbase['RGYEAR_gap'] * new_entbase['FINZB']
new_entbase['RGYEAR_mul_I'] = new_entbase['RGYEAR_gap'] * new_entbase['INUM']
new_entbase['RGYEAR_mul_MP'] = new_entbase['RGYEAR_gap'] * new_entbase['MPNUM']
new_entbase['RGYEAR_mul_FSTI'] = new_entbase['RGYEAR_gap'] * new_entbase['FSTINUM']
new_entbase['RGYEAR_mul_E'] = new_entbase['RGYEAR_gap'] * new_entbase['ENUM']


# In[576]:

# In[578]:

ZCZB_mul_YEAR_HandE = get_HandEforEntbase(new_entbase, 'ZCZB_mul_YEAR', len('ZCZB_mul_YEAR'))
MPNUM_HandE = get_HandEforEntbase(ZCZB_mul_YEAR_HandE, 'MPNUM', len('MPNUM'))
INUM_HandE = get_HandEforEntbase(MPNUM_HandE, 'INUM', len('INUM'))
FSTINUM_HandE = get_HandEforEntbase(INUM_HandE, 'FSTINUM', len('FSTINUM'))
RGYEAR_HandE = get_HandEforEntbase(FSTINUM_HandE, 'RGYEAR', len('RGYEAR'))


ZCZB_mul_YEAR_HandEandP  = get_HandEandPforEntbase(RGYEAR_HandE, 'ZCZB_mul_YEAR', len('ZCZB_mul_YEAR'))
MPNUM_HandEandP  = get_HandEandPforEntbase(ZCZB_mul_YEAR_HandEandP, 'MPNUM', len('MPNUM'))
INUM_HandEandP  = get_HandEandPforEntbase(MPNUM_HandEandP, 'INUM', len('INUM'))
FSTINUM_HandEandP  = get_HandEandPforEntbase(INUM_HandEandP, 'FSTINUM', len('FSTINUM'))
RGYEAR_HandEandP = get_HandEandPforEntbase(FSTINUM_HandEandP, 'RGYEAR', len('RGYEAR'))

#new_entbase2 = FSTINUM_HandE.copy()
new_entbase2 = RGYEAR_HandEandP.copy()


# In[580]:

entbaseDAE = new_entbase2.copy()


# In[581]:


train1 = pd.merge(new_train, entbaseDAE, on = 'EID', how = 'left')
test1 = pd.merge(test, entbaseDAE, on = 'EID', how = 'left')


# # Alter Feature

# In[583]:

alter = pd.read_csv(data_path + '2alter.csv')



import re


# In[588]:

from sklearn import preprocessing  

cdf = alter.copy()
label_enc = preprocessing.LabelEncoder()
label_enc.fit(alter.ALTERNO.values)
cdf.ALTERNO = label_enc.transform(alter.ALTERNO.values)
alterno_05, alterno_27 = label_enc.transform(['05', '27']) 
cdf['ALTDATE_LE'] = label_enc.fit_transform(alter.ALTDATE.values)
cdf['ALTDATE'] = cdf['ALTDATE'].apply(translateYear)

new_df = cdf.groupby(['EID'], as_index = 0)['ALTERNO'].count()
new_df.columns = ['EID', 'ALT_NUM']
ALTNO_VC = cdf.groupby(['EID'])['ALTERNO'].value_counts().unstack().fillna(0).add_prefix('ALTERNO_').reset_index()


# 12.4 alterno label feature
tmp = cdf.copy()

tmp_train = pd.merge(train[['TARGET', 'EID']], cdf[['ALTERNO','EID']], on = 'EID', how = 'left')
tmp_train.fillna(-1, inplace = True)

count_alter_label = tmp_train.groupby(['TARGET','ALTERNO']).count().reset_index()
alterno_label_0 = count_alter_label[count_alter_label.TARGET == 0]
alterno_label_1 = count_alter_label[count_alter_label.TARGET == 1]
alterno_label_0.rename(columns = {'EID': 'ALTERNO_label_0_count'},inplace = True)
alterno_label_1.rename(columns = {'EID': 'ALTERNO_label_1_count'},inplace = True)


alterno_label_count = pd.merge(alterno_label_0, alterno_label_1, on = 'ALTERNO', how = 'left')
alterno_label_count.drop(['TARGET_x', 'TARGET_y'], axis = 1, inplace = True)

alterno_label_count

def get_AlTERNO_LogOdds(row):
    label_0 = row.ALTERNO_label_0_count
    label_1 = row.ALTERNO_label_1_count
    all_count = label_0 + label_1
    c = 0.01
    logOdds = math.log(label_0 + c * label_0 / all_count) - math.log(label_1 + c * label_1 / all_count)
    return logOdds

alterno_label_count['ALTERNO_label_LogOdds'] = alterno_label_count.apply(get_AlTERNO_LogOdds, axis = 1)
alterno_label_count['ALTERNO_label_1_rate'] = alterno_label_count['ALTERNO_label_1_count'] * 1.0 / (alterno_label_count['ALTERNO_label_1_count']+ alterno_label_count['ALTERNO_label_0_count'])





# In[592]:

cdf['ALTGAP'] = cdf.ALTAF - cdf.ALTBE
cdf_05 = cdf[cdf.ALTERNO == alterno_05]
cdf_27 = cdf[cdf.ALTERNO == alterno_27]



# In[594]:

#version 2.2     ver4.1  add sum
cdf_05_be = cdf_05.groupby(['EID'])['ALTBE'].agg([sum, max, min, np.mean])
cdf_05_af = cdf_05.groupby(['EID'])['ALTAF'].agg([sum, max, min, np.mean])
cdf_27_be = cdf_27.groupby(['EID'])['ALTBE'].agg([sum, max, min, np.mean])
cdf_27_af = cdf_27.groupby(['EID'])['ALTAF'].agg([sum, max, min, np.mean])

cdf_05_count = cdf_05.groupby(['EID'])['ALTERNO'].count()
cdf_27_count = cdf_27.groupby(['EID'])['ALTERNO'].count()


# In[595]:

#ver4.2  add range
cdf_05_be['range'] = cdf_05_be['max'] - cdf_05_be['min']
cdf_05_af['range'] = cdf_05_af['max'] - cdf_05_af['min']
cdf_27_be['range'] = cdf_27_be['max'] - cdf_27_be['min']
cdf_27_af['range'] = cdf_05_be['max'] - cdf_27_af['min']


# In[596]:

cdf_05_be = cdf_05_be.add_prefix('ALTNO_05_BE_').reset_index()
cdf_05_af = cdf_05_af.add_prefix('ALTNO_05_AF_').reset_index()
cdf_27_be = cdf_27_be.add_prefix('ALTNO_27_BE_').reset_index()
cdf_27_af = cdf_27_af.add_prefix('ALTNO_27_AF_').reset_index()
cdf_05_count = cdf_05_count.reset_index().rename(columns = {'ALTERNO': 'ALTNO_05_Count'})
cdf_27_count = cdf_27_count.reset_index().rename(columns = {'ALTERNO': 'ALTNO_27_Count'})


# In[597]:

new_df1 = pd.merge(new_df, ALTNO_VC, on = 'EID', how = 'left')
new_df2 = pd.merge(new_df1, cdf_05_be, on = 'EID', how = 'left')
new_df3 = pd.merge(new_df2, cdf_05_af, on = 'EID', how = 'left')
new_df4 = pd.merge(new_df3, cdf_27_be, on = 'EID', how = 'left')
new_df5 = pd.merge(new_df4, cdf_27_af, on = 'EID', how = 'left')
new_df6 = pd.merge(new_df5, cdf_05_count, on = 'EID', how = 'left')
new_df7 = pd.merge(new_df6, cdf_27_count, on = 'EID', how = 'left')



# version 2   (效果好)

# In[599]:

# v1
"""
cdf_05_gap = cdf_05.groupby(['EID'])['ALTGAP'].agg({'max': np.max, 'min': np.min, 'mean': np.mean, 'median': np.median})
cdf_27_gap = cdf_27.groupby(['EID'])['ALTGAP'].agg({'max': np.max, 'min': np.min, 'mean': np.mean, 'median': np.median})"""
# version 2.2   v4  add sum
cdf_05_gap = cdf_05.groupby(['EID'])['ALTGAP'].agg([sum, max, min, np.mean])
cdf_27_gap = cdf_27.groupby(['EID'])['ALTGAP'].agg([sum, max, min, np.mean])

#v4.2  add range
cdf_05_gap['range'] = cdf_05_gap['max'] - cdf_05_gap['min']
cdf_27_gap['range'] = cdf_27_gap['max'] - cdf_27_gap['min']

cdf_05_gap = cdf_05_gap.add_prefix('ALTNO_05_GAP_').reset_index()
cdf_27_gap = cdf_27_gap.add_prefix('ALTNO_27_GAP_').reset_index()

new_df8 = pd.merge(new_df7, cdf_05_gap, on = 'EID', how = 'left')
new_df9 = pd.merge(new_df8, cdf_27_gap, on = 'EID', how = 'left')

# *** 
cdf2015 = cdf[cdf.ALTDATE >= 61]  # 61: 2015-01
cdf2015_alter_sum = cdf2015.groupby(['EID']).size().reset_index().rename(columns = {0: 'alter2015_sum'})

df_2015 = pd.merge(new_df, cdf2015_alter_sum, on='EID', how = 'left').fillna(0)
df_2015['alter2015_rate'] = df_2015['alter2015_sum'] / df_2015['ALT_NUM']
df_2015 = df_2015.drop(['ALT_NUM'], axis = 1)

new_df10 = pd.merge(new_df9, df_2015, on = 'EID', how = 'left')


# v4.1



# In[601]:

# 计算最近两次更改的时间差  数值越小代表越频繁
eid_altdate_df = pd.read_csv(data_path + 'eid_alterdate_gap.csv')
new_df11 = pd.merge(new_df10, eid_altdate_df, on = 'EID', how = 'left')


# v4.2

# In[602]:

alter_date_sta = cdf.groupby(['EID'])['ALTDATE'].agg([max, min]).add_prefix('alter_ALTDATE_').reset_index()
alter_date_sta['alter_ALTDATE_range'] = alter_date_sta['alter_ALTDATE_max'] - alter_date_sta['alter_ALTDATE_min']
new_df12 = pd.merge(new_df11, alter_date_sta, on = 'EID', how = 'left')


# In[603]:

# 12.4 alterno label feature
new_cdf = pd.merge(cdf, alterno_label_count, on = 'ALTERNO', how = 'left')
new_cdf.drop(['ALTERNO_label_0_count', 'ALTERNO_label_1_count'], axis = 1, inplace = True)

ALTERNO_avg_LogOdds = new_cdf.groupby(['EID'])['ALTERNO_label_LogOdds'].mean().reset_index()
ALTERNO_avg_label_1_rate = new_cdf.groupby(['EID'])['ALTERNO_label_1_rate'].mean().reset_index()

new_df13 = pd.merge(new_df12, ALTERNO_avg_LogOdds, on = 'EID', how ='left')
new_df14 = pd.merge(new_df13, ALTERNO_avg_label_1_rate, on = 'EID', how ='left')


# In[ ]:




# In[604]:

alterDAE = new_df14.copy()


# In[605]:

train2 = pd.merge(train1, alterDAE, on = 'EID', how = 'left')
test2 = pd.merge(test1, alterDAE, on = 'EID', how = 'left' )
train2.fillna(0, inplace = True)
test2.fillna(0, inplace = True)


# In[ ]:


# In[606]:

branch = pd.read_csv(data_path + '3branch.csv')



# In[611]:

branch['TYPECODE'] = branch['TYPECODE'].apply(lambda row: row[2:])


# In[613]:

new_branch = branch.groupby(['EID'])['IFHOME'].count()
new_branch = new_branch.reset_index().rename(columns = {'IFHOME': "Branch_Count"})

bstart_count = branch.groupby(['EID'])['B_REYEAR'].count().reset_index().rename(columns = {'B_REYEAR': "Bstart_count"})
bend_count = branch[branch.B_ENDYEAR.isnull() == False].groupby(['EID'])['B_ENDYEAR'].count().reset_index().rename(columns = {'B_ENDYEAR': 'Bend_count'})
bhome_count = branch.groupby(['EID', 'IFHOME'])['TYPECODE'].count().unstack().add_prefix('BRANCH_IFHOME_').add_suffix('_count').reset_index()

new_branch1 = pd.merge(new_branch, bstart_count, on = 'EID', how = 'left')
new_branch2 = pd.merge(new_branch1, bend_count, on = 'EID', how = 'left')
new_branch3 = pd.merge(new_branch2, bhome_count, on = 'EID', how = 'left')

new_branch3['BRANCH_IFHOME_0_rate'] = new_branch3['BRANCH_IFHOME_0_count'] / new_branch3['Branch_Count']
new_branch3['BRANCH_IFHOME_1_rate'] = new_branch3['BRANCH_IFHOME_1_count'] / new_branch3['Branch_Count']
new_branch3['Bend_rate'] = new_branch3['Bend_count'] / new_branch3['Branch_Count']

)

# In[615]:

#**
branch_start_2013 = branch[branch.B_REYEAR >= 2013].groupby(['EID']).size().reset_index().rename(columns = {0 : 'branch_start_from2013_count'})
branch_end_2013 = branch[branch.B_ENDYEAR >= 2013].groupby(['EID']).size().reset_index().rename(columns = {0 : 'branch_end_from2013_count'})

new_branch4 = pd.merge(new_branch3, branch_start_2013, on = 'EID', how = 'left')
new_branch5 = pd.merge(new_branch4, branch_end_2013, on = 'EID', how = 'left')

new_branch5.fillna(0, inplace = True)


# version3

# In[616]:

last_branch = branch[branch.B_ENDYEAR.isnull()]['B_REYEAR'].groupby(branch.EID).max().reset_index()
new_branch6 = pd.merge(new_branch5, last_branch, on = 'EID', how = 'left')

new_branch6.fillna(-1, inplace = True)


# ver4.2

# In[617]:

B_REYEAR_sta = branch['B_REYEAR'].groupby(branch.EID).agg([max, min])
B_REYEAR_sta['range'] = B_REYEAR_sta['max'] - B_REYEAR_sta['min']
B_REYEAR_sta = B_REYEAR_sta.add_prefix('branch_B_REYEAR_').reset_index()

B_ENDYEAR_sta = branch['B_ENDYEAR'].groupby(branch.EID).agg([max, min])
B_ENDYEAR_sta['range'] = B_ENDYEAR_sta['max'] - B_ENDYEAR_sta['min']
B_ENDYEAR_sta = B_ENDYEAR_sta.add_prefix('branch_B_ENDYEAR_').reset_index()

new_branch7 = pd.merge(new_branch6, B_REYEAR_sta, on = 'EID', how = 'left')
new_branch8 = pd.merge(new_branch7, B_ENDYEAR_sta, on = 'EID', how = 'left')




branchDAE = new_branch8.copy()


# In[620]:

train3 = pd.merge(train2, branchDAE, on = 'EID', how = 'left')
test3 = pd.merge(test2, branchDAE, on = 'EID', how = 'left')


# In[621]:


# In[622]:

invest = pd.read_csv(data_path + '4invest.csv')


# In[624]:

new_invest = invest.groupby(['EID']).size().reset_index().rename(columns = {0: 'invest_count'})
invest_home = invest.groupby(['EID', 'IFHOME'])['BTEID'].count().unstack().add_prefix('INVEST_IFHOME_').add_suffix('_count').reset_index()
#be_invested_count = invest.groupby(['BTEID']).size().reset_index().rename(columns = {0: 'be_invested_count', 'BTEID': 'EID'})
#be_invested_home = invest.groupby(['BTEID', 'IFHOME'])['EID'].count().unstack().add_prefix('BE_INVESTED_IFHOME_').add_suffix('_count').reset_index().rename(columns = {'BTEID': 'EID'})
invest_btbl = invest.groupby(['EID'])['BTBL'].agg([sum, max, min, np.mean, np.median]).add_prefix('invest_btbl_').reset_index()
invest_closed = invest[invest.BTENDYEAR.isnull() == False].groupby(['EID']).size().reset_index().rename(columns = {0: 'invest_closed_count'})

new_invest1 = pd.merge(new_invest, invest_home, on = 'EID', how = 'left')
new_invest4 = pd.merge(new_invest1, invest_btbl, on = 'EID', how = 'left')
new_invest5 = pd.merge(new_invest4, invest_closed, on = 'EID', how = 'left')

new_invest5['INVEST_IFHOME_0_rate'] = new_invest5['INVEST_IFHOME_0_count'] / new_invest5['invest_count']
new_invest5['INVEST_IFHOME_1_rate'] = new_invest5['INVEST_IFHOME_1_count'] / new_invest5['invest_count']
new_invest5['invest_closed_rate'] = new_invest5['invest_closed_count'] / new_invest5['invest_count']

new_invest5.fillna(0, inplace = True)


# version 2 .1

# In[625]:

#*
invest_2014_not_end = invest[invest.BTENDYEAR.isnull()]
invest_after_2014 = invest_2014_not_end[invest_2014_not_end.BTYEAR > 2014]
invest_after_2014_count = invest_after_2014.groupby(['EID']).size().reset_index().rename(columns = {0: 'invest_from2014_count'})
invest_btbl_after2014 = invest_after_2014.groupby(['EID'])['BTBL'].agg([max, min, np.mean]).add_prefix('invest_btbl_after_2014_').reset_index()

#**
invest_before_2008 = invest[invest.BTYEAR < 2008]
invest_before_2008_count = invest_before_2008.groupby(['EID']).size().reset_index().rename(columns = {0: 'invest_before2008_count'})
invest_btbl_before2008 = invest_before_2008.groupby(['EID'])['BTBL'].agg([max, min, np.mean]).add_prefix('invest_btbl_before_2008_').reset_index()


new_invest6 = pd.merge(new_invest5, invest_after_2014_count, on = 'EID', how = 'left')
new_invest7 = pd.merge(new_invest6, invest_btbl_after2014, on = 'EID', how = 'left')
new_invest8 = pd.merge(new_invest7, invest_before_2008_count, on = 'EID', how = 'left')
new_invest9 = pd.merge(new_invest8, invest_btbl_before2008, on = 'EID', how = 'left')

new_invest9.fillna(0, inplace = True)


# version 3.3

# In[626]:

investDAE = new_invest9.copy()




# In[628]:

train4 = pd.merge(train3, investDAE, on = 'EID', how = 'left')
test4 = pd.merge(test3, investDAE, on = 'EID', how = 'left')


# # Right Feature


# In[629]:

right = pd.read_csv(data_path + '5right.csv')


# 12.5 righttype label feature
tmp = right.copy()

tmp_train = pd.merge(train[['TARGET', 'EID']], tmp[['RIGHTTYPE','EID']], on = 'EID', how = 'left')
tmp_train.fillna(-1, inplace = True)

count_right_label = tmp_train.groupby(['TARGET','RIGHTTYPE']).count().reset_index()
righttype_label_0 = count_right_label[count_right_label.TARGET == 0]
righttype_label_1 = count_right_label[count_right_label.TARGET == 1]
righttype_label_0.rename(columns = {'EID': 'RIGHTTYPE_label_0_count'},inplace = True)
righttype_label_1.rename(columns = {'EID': 'RIGHTTYPE_label_1_count'},inplace = True)

count_right_label

righttype_label_count = pd.merge(righttype_label_0, righttype_label_1, on = 'RIGHTTYPE', how = 'left')
righttype_label_count.drop(['TARGET_x', 'TARGET_y'], axis = 1, inplace = True)


def get_RIGHTTYPE_LogOdds(row):
    label_0 = row.RIGHTTYPE_label_0_count
    label_1 = row.RIGHTTYPE_label_1_count
    all_count = label_0 + label_1
    c = 0.01
    logOdds = math.log(label_0 + c * label_0 / all_count) - math.log(label_1 + c * label_1 / all_count)
    return logOdds

righttype_label_count['RIGHTTYPE_label_LogOdds'] = righttype_label_count.apply(get_RIGHTTYPE_LogOdds, axis = 1)
righttype_label_count['RIGHTTYPE_label_1_rate'] = righttype_label_count['RIGHTTYPE_label_1_count'] * 1.0 / (righttype_label_count['RIGHTTYPE_label_1_count']+ righttype_label_count['RIGHTTYPE_label_0_count'])



# In[634]:

new_right = right.groupby(['EID']).size().reset_index().rename(columns = {0: 'right_count'})
right_type_count = right.groupby(['EID', 'RIGHTTYPE']).size().unstack().add_prefix('right_type_').add_suffix('_count').reset_index()

new_right1 = pd.merge(new_right, right_type_count, on = 'EID', how = 'left')

new_right1.fillna(0, inplace = True)




# In[ ]:




# version 2.1

# In[636]:

# ***
right_apply_after2013 = right[right.ASKDATE >= '2013-01']
right_apply_after2013_count = right_apply_after2013.groupby(['EID']).size().reset_index().rename(columns = {0: 'right_ask_after2013_count'})

# 筛选2013之后申请权利 且 获得权利的记录
right_get_after2013 = right_apply_after2013[right_apply_after2013.FBDATE.notnull()]
right_get_after2013_count = right_get_after2013.groupby(['EID']).size().reset_index().rename(columns = {0: 'right_get_after2013_count'})

right_after2013 = pd.merge(right_apply_after2013_count, right_get_after2013_count, on = 'EID', how = 'left')
right_after2013['right_get_after2013_rate'] = right_after2013['right_get_after2013_count'] / right_after2013['right_ask_after2013_count']

right_get_in2015_count = right[right.FBDATE >= '2015-01'].groupby(['EID']).size().reset_index().rename(columns = {0: 'right_get_in2015_count'})

new_right2 = pd.merge(new_right1, right_after2013, on = 'EID', how = 'left')
new_right3 = pd.merge(new_right2, right_get_in2015_count, on = 'EID', how = 'left')
new_right3.fillna(0, inplace = True)


# v4.2

# In[637]:

rightDATE = right[['EID']]
rightDATE['ASKDATE'] = right.ASKDATE.apply(translateYear)
rightDATE['FBDATE'] = right.FBDATE.apply(translateYear)

ASKDATE_sta = rightDATE['ASKDATE'].groupby(rightDATE.EID).agg([max, min])
ASKDATE_sta['range'] = ASKDATE_sta['max'] - ASKDATE_sta['min']
ASKDATE_sta = ASKDATE_sta.add_prefix('right_ASKDATE_').reset_index()

FBDATE_sta = rightDATE['FBDATE'].groupby(rightDATE.EID).agg([max, min])
FBDATE_sta['range'] = FBDATE_sta['max'] - FBDATE_sta['min']
FBDATE_sta = FBDATE_sta.add_prefix('right_FBDATE_').reset_index()

new_right4 = pd.merge(new_right3, ASKDATE_sta, on = 'EID', how = 'left')
new_right5 = pd.merge(new_right4, FBDATE_sta, on = 'EID', how = 'left')


# In[639]:

rightDAE = new_right5.copy()


# In[640]:

train5 = pd.merge(train4, rightDAE , on = 'EID', how = 'left')
test5 = pd.merge(test4, rightDAE , on = 'EID', how = 'left')


# # Project Feature

# In[641]:

project = pd.read_csv(data_path + '6project.csv')



new_project = project.groupby(['EID']).size().reset_index().rename(columns = {0: 'project_count'})
project_home = project.groupby(['EID', 'IFHOME'])['TYPECODE'].count().unstack().add_prefix('PROJECT_IFHOME_').add_suffix('_count').reset_index()
last_project = project['DJDATE'].groupby(project.EID).max().reset_index()
last_project['DJDATE'] = last_project['DJDATE'].apply(translateYear)



# In[645]:

new_project1 = pd.merge(new_project, project_home, on = 'EID', how = 'left')
new_project2 = pd.merge(new_project1, last_project, on = 'EID', how = 'left')

new_project2.fillna(0, inplace = True)


# version 2

# In[646]:

# *
project_in2015_count = project[project.DJDATE >= '2015-01'].groupby(['EID']).size().reset_index().rename(columns = {0: 'project_in2015_count'})

new_project3 = pd.merge(new_project2, project_in2015_count, on = 'EID', how = 'left')
new_project3.fillna(0, inplace = True)


# In[647]:

projectDAE = new_project3.copy()


# In[648]:

train6 = pd.merge(train5, projectDAE, on = 'EID', how = 'left')
test6 = pd.merge(test5, projectDAE, on = 'EID', how = 'left')


# # Lawsuit Feature

# In[649]:

lawsuit = pd.read_csv(data_path + '7lawsuit.csv')




# In[652]:

new_lawsuit = lawsuit.groupby(['EID']).size().reset_index().rename(columns = {0: 'lawsuit_count'})

lawsuit_sta = lawsuit.groupby(['EID']).LAWAMOUNT.agg([sum, max, min, np.mean, np.median])
lawsuit_sta['gap'] = lawsuit_sta['max'] -  lawsuit_sta['min']
lawsuit_sta = lawsuit_sta.add_prefix('lawsuit_LAWAMOUNT_').reset_index()
lawsuit_date = lawsuit['LAWDATE'].groupby(lawsuit.EID).agg([max, min]).add_prefix('lawsuit_LAWDATE_').reset_index()

lawsuit_date['lawsuit_LAWDATE_max'] = lawsuit_date['lawsuit_LAWDATE_max'].apply(translateYear)
lawsuit_date['lawsuit_LAWDATE_min'] = lawsuit_date['lawsuit_LAWDATE_min'].apply(translateYear)
lawsuit_date['lawsuit_LAWDATE_gap'] = lawsuit_date['lawsuit_LAWDATE_max'] - lawsuit_date['lawsuit_LAWDATE_min']

new_lawsuit1 = pd.merge(new_lawsuit, lawsuit_sta, on = 'EID', how = 'left')
new_lawsuit2 = pd.merge(new_lawsuit1, lawsuit_date, on = 'EID', how = 'left')


# version 2.1

# In[653]:

lawsuit_in2015 = lawsuit[lawsuit.LAWDATE >= '2015-01-01']
lawsuit_in2015_count = lawsuit_in2015.groupby(['EID']).size().reset_index().rename(columns = {0: 'lawsuit_in2015_count'})
lawsuit_in2015_sta = lawsuit_in2015.groupby(['EID'])['LAWAMOUNT'].agg([sum, max, min, np.mean])
lawsuit_in2015_sta['gap'] = lawsuit_in2015_sta['max'] -  lawsuit_in2015_sta['min']
lawsuit_in2015_sta = lawsuit_in2015_sta.add_prefix('lawsuit_LAWAMOUNT_2015_').reset_index()

new_lawsuit3 = pd.merge(new_lawsuit2, lawsuit_in2015_count, on = 'EID', how = 'left')
new_lawsuit4 = pd.merge(new_lawsuit3, lawsuit_in2015_sta, on = 'EID', how = 'left')
new_lawsuit4.fillna(0, inplace = True)




# In[655]:

lawsuitDAE = new_lawsuit4.copy()


# In[656]:

train7 = pd.merge(train6, lawsuitDAE, on = 'EID', how = 'left')
test7 = pd.merge(test6, lawsuitDAE, on = 'EID', how = 'left')


# In[ ]:




# # Breakfaith Feautre

# In[657]:

breakfaith = pd.read_csv(data_path + '8breakfaith.csv')



# In[659]:

new_breakfaith = breakfaith.groupby(breakfaith.EID).size().reset_index().rename(columns = {0: 'breakfaith_count'})
not_end_breakfaith = breakfaith[breakfaith.SXENDDATE.isnull()].groupby(breakfaith.EID).size().reset_index().rename(columns = {0: 'not_end_breakfaith_count'})
breakfaith_date = breakfaith.groupby(breakfaith.EID)['FBDATE'].agg([max, min]).add_prefix('breakfaith_date_').reset_index()
breakfaith_date['breakfaith_date_max'] = breakfaith_date['breakfaith_date_max'].apply(translateYearFB)
breakfaith_date['breakfaith_date_min'] = breakfaith_date['breakfaith_date_min'].apply(translateYearFB)
breakfaith_date['breakfaith_date_gap'] = breakfaith_date['breakfaith_date_max'] - breakfaith_date['breakfaith_date_min']


new_breakfaith1 = pd.merge(new_breakfaith, not_end_breakfaith, on = 'EID', how = 'left')
new_breakfaith2 = pd.merge(new_breakfaith1, breakfaith_date, on = 'EID', how = 'left')
new_breakfaith2['not_end_breakfaith_rate'] = new_breakfaith2['not_end_breakfaith_count'] / new_breakfaith2['breakfaith_count']

# rate=0 代表所有失信记录全部结束 1 代表全部没结束
new_breakfaith2.fillna(0, inplace = True)



# In[660]:

breakfaithDAE = new_breakfaith2.copy()


# In[661]:

train8 = pd.merge(train7, breakfaithDAE, on = 'EID', how = 'left')
test8 = pd.merge(test7, breakfaithDAE, on = 'EID', how = 'left')



# # Recruit Feature

# In[663]:

recruit = pd.read_csv(data_path + '9recruit.csv')

def cleanPNUM(df):
    data = df['PNUM'].fillna('0').values
    pnum = []
    for row in data:
        if '人' in row:
            #print row
            row = int(row[:-1])
        elif row!='若干':
            row = int(row)
        elif row=='若干':
            row = 4.7
        pnum.append(row)

    df['PNUM'] = pnum
    return df


# In[107]:

recruit = cleanPNUM(recruit)


# In[ ]:




new_recurit = recruit.groupby(recruit.EID).size().reset_index().rename(columns = {0: 'recurit_count'})

recruit_type = recruit.groupby(['EID', 'WZCODE'])['PNUM'].sum().unstack().add_prefix('WZCODE_').add_suffix('_count').reset_index()
recruit_type.fillna(0, inplace = True)
recurit_2015 = recruit[recruit.RECDATE >= '2015-01']
recurit_2015_count = recurit_2015.groupby(recurit_2015.EID)['PNUM'].sum().reset_index().rename(columns = {0: 'recurit2015_count'})

new_recurit1 = pd.merge(new_recurit,recruit_type , on = 'EID', how = 'left')
new_recurit2 = pd.merge(new_recurit1,recurit_2015_count , on = 'EID', how = 'left')

new_recurit2.fillna(0, inplace = True)


# version 2 （有效果）

# In[668]:

recruit_type1 = pd.DataFrame(recruit_type['WZCODE_zp01_count'] +  recruit_type['WZCODE_zp02_count'] +  recruit_type['WZCODE_zp03_count'], columns = ['recurit_sum'])

# **
recruit_type1['EID'] = recruit_type.EID
recruit_type1['zp01_rate'] =  recruit_type['WZCODE_zp01_count'] / recruit_type1['recurit_sum']
recruit_type1['zp02_rate'] =  recruit_type['WZCODE_zp02_count'] / recruit_type1['recurit_sum']
recruit_type1['zp03_rate'] =  recruit_type['WZCODE_zp03_count'] / recruit_type1['recurit_sum']
recruit_type1.fillna(0, inplace = True)

new_recurit3 = pd.merge(new_recurit2, recruit_type1 , on = 'EID', how = 'left')


# v3.4

# In[669]:

"""new_recurit4 = pd.merge(new_recurit3, EID_HY_ETYPE, on = 'EID', how = 'left')

prefix_index = -3
new_recurit5 = get_HandE(new_recurit4, 'recurit_sum', prefix_index)

new_recurit5.drop(['ETYPE', 'HY'], axis = 1, inplace = True)"""


# v4.4

# In[670]:

recurit_newdate = recruit.copy()
recurit_newdate.RECDATE = recurit_newdate.RECDATE.apply(translateYearRECRUIT)


RECDATE_sta = recurit_newdate['RECDATE'].groupby(recurit_newdate.EID).agg([max, min])
RECDATE_sta['range'] = RECDATE_sta['max'] - RECDATE_sta['min']
RECDATE_sta = RECDATE_sta.add_prefix('recruit_RECDATE_').reset_index()

new_recurit4 = pd.merge(new_recurit3, RECDATE_sta , on = 'EID', how = 'left')


# Merge Train and Test

# In[671]:

recruitDAE = new_recurit4.copy()


# In[672]:

train9 = pd.merge(train8, recruitDAE, on = 'EID', how = 'left')
test9 = pd.merge(test8, recruitDAE, on = 'EID', how = 'left')



# # Entbase_Alter Feature

# version 3

# In[674]:

ent_alter = pd.merge(cdf, entbase, on = 'EID', how ='left')
# HY ALTER 均值与差值

ent_alter_gb_HY = ent_alter.groupby(['HY']).size().reset_index().rename(columns = {0: 'ent_alter_HY_alterno_count'})
HY_EID_count = entbase.groupby(['HY']).size().reset_index().rename(columns = {0: 'entbase_HY_EID_count'})

ent_alter_gb_HY1 = pd.merge(ent_alter_gb_HY, HY_EID_count, on = 'HY', how = 'left')
ent_alter_gb_HY1['ent_alter_HY_alterno_mean'] = ent_alter_gb_HY1['ent_alter_HY_alterno_count'] / ent_alter_gb_HY1['entbase_HY_EID_count']
ent_alter_gb_HY2 = ent_alter_gb_HY1.drop(['ent_alter_HY_alterno_count' , 'entbase_HY_EID_count'], axis = 1)


# ETYPE ALTER 均值与差值
ent_alter_gb_ETYPE = ent_alter.groupby(['ETYPE']).size().reset_index().rename(columns = {0: 'ent_alter_ETYPE_alterno_count'})
ETYPE_EID_count = entbase.groupby(['ETYPE']).size().reset_index().rename(columns = {0: 'entbase_ETYPE_EID_count'})

ent_alter_gb_ETYPE1 = pd.merge(ent_alter_gb_ETYPE, ETYPE_EID_count, on = 'ETYPE', how = 'left')
ent_alter_gb_ETYPE1['ent_alter_ETYPE_alterno_mean'] = ent_alter_gb_ETYPE1['ent_alter_ETYPE_alterno_count'] / ent_alter_gb_ETYPE1['entbase_ETYPE_EID_count']
ent_alter_gb_ETYPE2 = ent_alter_gb_ETYPE1.drop(['ent_alter_ETYPE_alterno_count' , 'entbase_ETYPE_EID_count'], axis = 1)

ent_alter_gb_HY_ETYPE = ent_alter.groupby(['ETYPE', 'HY']).size().reset_index().rename(columns = {0: 'ent_alter_ETYPE_alterno_count'})
ETYPE_EID_count = entbase.groupby(['ETYPE']).size().reset_index().rename(columns = {0: 'entbase_ETYPE_EID_count'})

ent_alter_gb_ETYPE1 = pd.merge(ent_alter_gb_ETYPE, ETYPE_EID_count, on = 'ETYPE', how = 'left')
ent_alter_gb_ETYPE1['ent_alter_ETYPE_alterno_mean'] = ent_alter_gb_ETYPE1['ent_alter_ETYPE_alterno_count'] / ent_alter_gb_ETYPE1['entbase_ETYPE_EID_count']
ent_alter_gb_ETYPE2 = ent_alter_gb_ETYPE1.drop(['ent_alter_ETYPE_alterno_count' , 'entbase_ETYPE_EID_count'], axis = 1)


# In[675]:

train10 = pd.merge(train9, ent_alter_gb_HY2, on = 'HY', how = 'left')
train11 = pd.merge(train10, ent_alter_gb_ETYPE2, on = 'ETYPE', how = 'left')
train11.ent_alter_ETYPE_alterno_mean.fillna(0, inplace = True)
train11.ent_alter_HY_alterno_mean.fillna(0, inplace = True)
train11['ent_alter_HY_alterno_gap'] = train11['ALT_NUM'] - train11['ent_alter_HY_alterno_mean']
train11['ent_alter_ETYPE_alterno_gap'] = train11['ALT_NUM'] - train11['ent_alter_ETYPE_alterno_mean']

test10 = pd.merge(test9, ent_alter_gb_HY2, on = 'HY', how = 'left')
test11= pd.merge(test10, ent_alter_gb_ETYPE2, on = 'ETYPE', how = 'left')
test11.ent_alter_ETYPE_alterno_mean.fillna(0, inplace = True)
test11.ent_alter_HY_alterno_mean.fillna(0, inplace = True)
test11['ent_alter_HY_alterno_gap'] = test11['ALT_NUM'] - test11['ent_alter_HY_alterno_mean']
test11['ent_alter_ETYPE_alterno_gap'] = test11['ALT_NUM'] - test11['ent_alter_ETYPE_alterno_mean']


# # Qualification

# In[676]:

qualification = pd.read_csv(data_path + '10qualification.csv')


# In[677]:

def translateYearQualification(date):
    if type(date) == float:
        return np.nan
    year = int(date[:4])
    month = int(date[-2:])
    return (year-2007)*12 + month


# In[678]:

qualification_count = qualification.groupby(qualification.EID).size().reset_index().rename(columns = {0: 'qualification_count'})

new_qualification = qualification.copy()
new_qualification['BEGINDATE'] = new_qualification.BEGINDATE.apply(translateYearQualification)
new_qualification['EXPIRYDATE'] = new_qualification.EXPIRYDATE.apply(translateYearQualification)

qualification_type = new_qualification.groupby(['EID', 'ADDTYPE']).size().unstack().add_prefix('qualification_type_').add_suffix('_count').reset_index()
qualification_type.fillna(0, inplace = True)

qualification_bdate_sta = new_qualification.groupby(new_qualification.EID).BEGINDATE.agg([max, min]).add_prefix('Qua_Bdate_').reset_index()
qualification_bdate_sta['Qua_Bdate_range'] = qualification_bdate_sta['Qua_Bdate_max'] - qualification_bdate_sta['Qua_Bdate_min']
qualification_edate_sta = new_qualification.groupby(new_qualification.EID).EXPIRYDATE.agg([max, min]).add_prefix('Qua_Edate_').reset_index()

new_qualification2 = pd.merge(qualification_count, qualification_type, on = 'EID', how = 'left')
new_qualification3 = pd.merge(new_qualification2, qualification_bdate_sta, on = 'EID', how = 'left')
new_qualification4 = pd.merge(new_qualification3, qualification_edate_sta, on = 'EID', how = 'left')


# In[679]:

qualificationDAE = new_qualification4.copy()


# In[680]:

train12 = pd.merge(train11, qualificationDAE, on = 'EID', how = 'left')
test12 = pd.merge(test11, qualificationDAE, on = 'EID', how = 'left')


# In[681]:

import copy
def get_nan_info(nan_train):
    tmp = copy.deepcopy(nan_train)
    tmp.fillna(-999, inplace = True)
    tmp['n_null'] = (tmp == -999).sum(axis=1)
    tmp['discret_null'] = tmp.n_null
    tmp.discret_null[tmp.discret_null<=40] = 1
    tmp.discret_null[(tmp.discret_null>40)&(tmp.discret_null<=73)] = 2
    tmp.discret_null[(tmp.discret_null>73)&(tmp.discret_null<=82)] = 3
    tmp.discret_null[(tmp.discret_null>82)&(tmp.discret_null<=85)] = 4
    tmp.discret_null[(tmp.discret_null>85)&(tmp.discret_null<=91)] = 5
    tmp.discret_null[(tmp.discret_null>91)&(tmp.discret_null<=100)] = 6
    tmp.discret_null[tmp.discret_null>100] = 7
    return tmp


# In[682]:

temp = copy.deepcopy(train12)
temp1 = get_nan_info(temp)
train13 = pd.concat([train12, temp1[['discret_null']] ], axis = 1)

temp2 = copy.deepcopy(test12)
temp3 = get_nan_info(temp2)
test13 = pd.concat([test12, temp3[['discret_null']] ], axis = 1)


# # Save Feature (Remember change the file name)

# In[683]:

#train13.drop(['ENDDATE_n'], axis = 1, inplace = True)


# In[684]:

train13['len_EID_NUMBER'] = train13['EID_NUMBER'].apply(lambda x:int(x/20))
train13 = train13.drop(['EID_NUMBER'], axis=1)
test13['len_EID_NUMBER'] = test13['EID_NUMBER'].apply(lambda x:int(x/20))
test13 = test13.drop(['EID_NUMBER'], axis=1)

savetrain = train13.copy()
savetest = test13.copy()


# In[ ]:


savetrain.to_csv(feature_path + 'train_v1.8.csv', index = False)
savetest.to_csv(feature_path + 'test_v1.8.csv', index = False)


# In[696]:

#savetrain.to_csv(feature_path + 'train_v1.2.csv', index = False)
#savetest.to_csv(feature_path + 'test_v1.2.csv', index = False)

