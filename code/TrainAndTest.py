import math
import sys
import re
import time
import pandas as pd
import numpy as np
import xgboost as xgb

def log(info):
    print (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), info)

class MLframe(object):
    """docstring for MLframe"""

    def __init__(self, train_x, train_y, test_x, seed=0):
        self.seed = seed
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x

    def stacking(self, clf, nfold):
        oof_train = np.zeros([self.train_x.shape[0], 1])
        oof_test = np.zeros([self.test_x.shape[0], 1])
        oof_test_skf = np.empty([self.test_x.shape[0], nfold])
        skf = list(KFold(n_splits=nfold, random_state=self.seed).split(self.train_x, self.train_y))
        for i, (train_index, test_index) in enumerate(skf):
            LogInfo("--fold:" + str(i))
            x_tr = self.train_x[train_index, :]
            y_tr = self.train_y[train_index]
            x_te = self.train_x[test_index, :]
            print (x_tr.shape, x_te.shape)

            clf.train(x_tr, y_tr)

            oof_train[test_index, 0] = clf.predict(x_te)
            oof_test_skf[:, i] = clf.predict(self.test_x)

        evalu = evaluate(oof_train[:, 0], self.train_y)
        print (evalu.mape())
        oof_test[:] = oof_test_skf.mean(axis=1).reshape(self.test_x.shape[0], 1)
        return oof_train, oof_test

    def train_test(self, clf, test_y, isxgb=False):
        if isxgb == True:
            clf.train_test(self.train_x, self.train_y, self.test_x, test_y)
        else:
            clf.train(self.train_x, self.train_y)
            predict = clf.predict(self.test_x)
            evalu = evaluate(predict, test_y)
            print (evalu.mape())

    def train_predict(self, clf):
        clf.train(self.train_x, self.train_y)
        predict = clf.predict(self.test_x)
        return predict

    def get_importance(self, clf, featureName):
        print (featureName[0:])
        clf.train(self.train_x, self.train_y)
        imp_f = clf.feature_importances()
        df = pd.DataFrame({'feature': [featureName[int(key[1:])] for key, value in imp_f.items()],
                           'fscore': [value for key, value in imp_f.items()]})
        # mean_imp = np.mean(df['fscore'])
        # top_f = df.loc[df['fscore'] >= mean_imp]
        df['fscore'] = df['fscore'] / df['fscore'].sum()
        df = df.sort_values(by='fscore', ascending=False)
        print (df)


class evaluate(object):
    """docstring for evaluate"""

    def __init__(self, predict, truth):
        '''
        predict:row_id, shop_id, probability
        truth: row_id ,shop_id
        '''
        self.predict = predict
        self.true = truth

    def selfDefinePrecision(self):
        '''
        '''
        pass


class XgbWrapper(object):
    def __init__(self, seed=0, params=None, rounds=100, has_weight=False):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = rounds
        self.has_weight = has_weight

    def train(self, x_train, y_train, rounds = 100):
        if self.has_weight == True:
            print ('has weight...')
            weight = get_weight(y_train)
            dtrain = xgb.DMatrix(x_train, label=y_train, weight=weight)
        else:
            dtrain = xgb.DMatrix(x_train, label=y_train)
        watchlist = [(dtrain, 'train')]
        if rounds == 100:
            self.gbdt = xgb.train(self.param, dtrain, self.nrounds, watchlist, verbose_eval=20)
        else:
            self.gbdt = xgb.train(self.param, dtrain, rounds, watchlist, verbose_eval=20)
        # return self.gbdt

    def train_test(self, x_train, y_train, x_test, y_test):
        if self.has_weight == True:
            print ('has weight...')
            weight = get_weight(y_train)
            # print weight
            dtrain = xgb.DMatrix(x_train, label=y_train, weight=weight)
        else:
            dtrain = xgb.DMatrix(x_train, label=y_train)
        dtest = xgb.DMatrix(x_test, label=y_test)
        watchlist = [(dtrain, 'train'), (dtest, 'dtest')]
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds, watchlist, verbose_eval=1)
        # return self.gbdt

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))

    def feature_importances(self):
        return self.gbdt.get_fscore()

    def default_cv(self, x_train, y_train):
        if self.has_weight == True:
            print ('has weight...')
            weight = get_weight(y_train)
            dtrain = xgb.DMatrix(x_train, label=y_train, weight=weight)
        else:
            dtrain = xgb.DMatrix(x_train, label=y_train)
        res = xgb.cv(self.param, dtrain, self.nrounds, early_stopping_rounds = 100, verbose_eval=20)
        best_round = len(res)
        best_score = res.tail(1)["test-auc-mean"].values[0]
        return res

# In[2]:

xgb_params = {
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'stratified':True,
    'max_depth':8,
    'min_child_weight':0.5,
    'gamma':2,
    'subsample':0.8,
    'colsample_bytree':0.8,
    
    #'lambda':0.001,   #550
#     'alpha':0.00001,
#     'lambda_bias':0.1,
    #'threads':512,
    'eta': 0.02,
    'seed':42, 
    'silent': 1
}
xgb_config = {
    'round' : 2000,
    'folds' : 5
}


# In[3]:
# lab

version = '1.13.1'
feature_importance_version = '1.13.1_d8_gamma2_r300_t0_'
throw_threshold = 0


data_path = "../data/"
result_path =  "../data/result/"
feature_path =  "../data/feature/"
analysis_path =  "../data/analysis/"
function_path =  "../data/function/"

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
train = pd.read_csv(feature_path + 'train_v' + str(version) + '.csv')
test = pd.read_csv(feature_path + 'test_v' + str(version) + '.csv')
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# In[14]:
#train_drop_column = ['EID','TARGET']
#test_drop_column = ['EID']
train_drop_column = ['TARGET', 'EID', 'ENDDATE_n', 'n_null']
test_drop_column = ['EID', 'n_null']

####  with one hot #########

#train_drop_column.extend(['HY', 'ETYPE'])
#test_drop_column.extend(['HY', 'ETYPE'])

############################



######################### throw col #######################
#v2.2 0.701678
#v3 50 0.70318
"""
import pickle
with open('fscore_version' + str(feature_importance_version) + '.pickle', 'rb') as handle:
    fs = pickle.load(handle)

throw_col = []
throw_threshold = 5
for i in fs.items():
    if i[1] <= throw_threshold:
        key = i[0]
        if key[0:3] == 'HY_' or key[0:6] == 'ETYPE_' or key[0:8] == 'ALTERNO_':
            continue
        else:
            throw_col.append(i[0])

train_drop_column.extend(throw_col)
test_drop_column.extend(throw_col)

print (throw_col)
print ('\n')
print ('throw_threshold:' + str(throw_threshold) + '\n')
"""
############################################################

train_x = train.drop(train_drop_column, axis = 1)
train_y = train.TARGET.values

test_ID = test.EID.values 
test_x = test.drop(test_drop_column, axis = 1)


xgb_params['scale_pos_weights '] = (float)(len(train_y[train_y == 0]))/len(train_y[train_y == 1])
xgbModel = XgbWrapper(xgb_params['seed'], xgb_params, xgb_config['round'])



info = '_d'+str(xgb_params['max_depth'])+ '_gamma' + str(xgb_params['gamma']) +\
         '_r300_t' + str(throw_threshold) + '_' 

log('version' + str(version) + info + ', with round ' + str(xgb_config['round']))

res_cv = xgbModel.default_cv(train_x, train_y)
best_round = len(res_cv) + 300
best_score = res_cv.tail(1)["test-auc-mean"].values[0]

print ('best round:' + str(best_round))
print ('best score: ' + str(best_score))




xgbModel.train(train_x, train_y, best_round)




feature_importances = xgbModel.feature_importances()
import pickle
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

with open('fscore_version' + str(version) + info + '.pickle', 'wb') as handle:
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    pickle.dump(feature_importances, handle, protocol=pickle.HIGHEST_PROTOCOL)





res = xgbModel.predict(test_x)
sub = pd.DataFrame({'EID':test_ID, 'FORTARGET':[1 if i > 0.219 else 0 for i in res], 'PROB':res})
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
sub.to_csv(result_path + 'res_v' + str(version) + info + '_' + str(best_score) + '.csv', index= False)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

"""
best_round = 794

xgbModel.train(train_x, train_y, best_round - 100)
res = xgbModel.predict(test_x)
sub = pd.DataFrame({'EID':test_ID, 'FORTARGET':[1 if i > 0.219 else 0 for i in res], 'PROB':res})
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
sub.to_csv(result_path + 'res_v' + str(version) + info + '_' + '_200round.csv', index= False)
# In[ ]:
"""


