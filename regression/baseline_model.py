# coding:utf-8
# Created by chen on 14/09/2018
# email: q.chen@student.utwente.nl
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

def load_data_kfold(k,X,Y):
    ''''
    cross validation
    '''
    folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=1).split(X,Y))
    return folds

def base_line_model_cal(weekNum):
    '''
    :param weekNum:  how many weeks after nurse assessment
    :return: nurse assessment as a baseline model, calculate the rmse, mae and r2, use the same 5-cross validation
    '''
    pandas2ri.activate()
    readRDS = robjects.r['readRDS']
    fileName = "../data/allCom_v"+str(weekNum)+"_baseLine.rds"
    df = readRDS(fileName)
    df = pandas2ri.ri2py(df)

    df = df.rename({'duration_next_week':'label'}, axis='columns')
    preds  = df['label']
    df = df.drop('label',axis=1)
    # change dataframe into numpy
    X = df['Estimated_Minutes_Weekly'].values
    y = preds.values

    k = 5
    folds = load_data_kfold(k, X, y)
    resultsTemp =  {}

    for j, (train_idx, val_idx) in enumerate(folds):

        X_valid_cv = X[val_idx]
        y_valid_cv = y[val_idx]

        n = len(y_valid_cv)
        mae = sum(abs(X_valid_cv - y_valid_cv))/n
        mean_of_oberve = X_valid_cv.mean()
        ssTOT = sum(np.power((X_valid_cv-mean_of_oberve),2))
        ssRes = sum(np.power((X_valid_cv-y_valid_cv),2))
        R_squared = 1- (ssRes/ssTOT)
        RMSE = np.sqrt(ssRes/n)

        resultsTemp.setdefault("MAE_week_" + str(weekNum), []).append(mae)
        resultsTemp.setdefault("R2_week_"+str(weekNum), []).append(R_squared)
        resultsTemp.setdefault("RMSE_week_"+ str(weekNum), []).append(RMSE)

    resultsTemp["MAE_week_" + str(weekNum)+"_average"] = np.mean(resultsTemp["MAE_week_" + str(weekNum)])
    resultsTemp["MAE_week_" + str(weekNum)+"_variance"] = np.var(resultsTemp["MAE_week_" + str(weekNum)])

    resultsTemp["R2_week_" + str(weekNum) + "_average"] = np.mean(resultsTemp["R2_week_" + str(weekNum)])
    resultsTemp["R2_week_" + str(weekNum) + "_variance"] = np.var(resultsTemp["R2_week_" + str(weekNum)])

    resultsTemp["RMSE_week_" + str(weekNum) + "_average"] = np.mean(resultsTemp["RMSE_week_" + str(weekNum)])
    resultsTemp["RMSE_week_" + str(weekNum) + "_variance"] = np.var(resultsTemp["RMSE_week_" + str(weekNum)])
    return resultsTemp

results = {}
## save from 1 to 15 weeks after nurse assessment
for i in range(15):
    resultTemp =base_line_model_cal(i+1)
    if bool(results):
        results.update(resultTemp)
    else:
        results = resultTemp.copy()

## save result into files
fileName = "baseline_performance.txt"
with open(fileName, "w") as file:
    for key, value in results.items():
        file.write("%s:%s\n" % (key, value))