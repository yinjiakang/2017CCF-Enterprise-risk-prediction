import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from dateutil import parser
from datetime import datetime


data_path = "../../data/"
result_path =  "../../data/result/"
feature_path =  "../../data/feature/"
analysis_path =  "../../data/analysis/"
function_path =  "../../data/function/"

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

import math
def translateYear(date):
    if type(date) == float:
        return np.nan
    year = int(date[:4])
    month = int(date[-2:])
    return (year-2015)*12 + month

# 计算最近两次更改的时间差
def get_eid_alterdate(alter):
    alter['ALTDATE'] = alter['ALTDATE'].apply(translateYear)

    eid_altdate = alter[['EID', 'ALTDATE']].values
    last_alter_list = []
    second_alter_list = []
    eid_list = []

    for line in eid_altdate:
        eid, date = line[0], line[1]
        if eid in eid_list:
            idx = eid_list.index(eid)
            if date >= last_alter_list[idx]:
                second_alter_list[idx] = last_alter_list[idx]
                last_alter_list[idx] = date
            elif date > second_alter_list[idx]:
                second_alter_list[idx] = date
        else:
            eid_list.append(eid)
            last_alter_list.append(date)
            second_alter_list.append(0)

    altdate_gap_list = []
    for i in range(len(eid_list)):
        altdate_gap_list.append(last_alter_list[i] - second_alter_list[i])
    eid_altdate_df = pd.DataFrame({'EID': eid_list, 'ALTDATE_latest_two_gap': altdate_gap_list})
    eid_altdate_df.to_csv(data_path + 'eid_alterdate_gap.csv', index = False)



if __name__ == '__main__':
    get_eid_alterdate(alter)