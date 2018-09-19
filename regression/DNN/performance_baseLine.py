# coding:utf-8
# Created by chen on 19/09/2018
# email: q.chen@student.utwente.nl
'''
test wehether DNN model perform better than baseline
'''
from tools import *
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import numpy as np
import pandas as pd

def plot_base_and_dnn(weekNum, kFold):
    fileName = "../../data/allCom_v"+str(weekNum)+"_final.rds"
    pandas2ri.activate()
    readRDS = robjects.r['readRDS']
    df = readRDS(fileName)
    df = pandas2ri.ri2py(df)
    # df = pd.read_csv("C:/work/ecare/data/allCom_v1_final.csv")
    df = df.rename({'duration_next_week':'label'}, axis='columns')

    preds  = df['label']
    df = df.drop('label',axis=1)
    X = df.values
    y = preds.values

    results = []
    for j in range(kFold):
        name_weights = "model/v"+str(weekNum)+"/final_model_fold" + str(j) + "_weights.h5"
        model2 = load_model(name_weights,custom_objects={'rmse':rmse, 'r_square':r_square})
        y_pred = model2.predict(X)
        results.append(y_pred)


    Y_pred_mean = [sum(x)/kFold for x in zip(*results)]
    Y_pred_mean = [x[0] for x in Y_pred_mean]

    pandas2ri.activate()
    readRDS = robjects.r['readRDS']
    fileName = "../../data/allCom_v" + str(weekNum) + "_baseLine.rds"
    df_base_line = readRDS(fileName)
    df_base_line = pandas2ri.ri2py(df_base_line)

    df_base_line['prediction'] = pd.Series(Y_pred_mean)
    df_base_line['prediction_true_y'] = pd.Series(y)

    return df_base_line

## plot for data that one week after nurse assessment
df_base_line = plot_base_and_dnn(1,5)

## plot
import matplotlib.pyplot as plt
plt.figure()
ax = df_base_line.plot.scatter(x='Estimated_Minutes_Weekly', y='duration_next_week', color='DarkBlue',alpha=0.3,
                               label='nurse estimation')

df_base_line.plot.scatter(x='prediction', y='prediction_true_y', color='DarkGreen', alpha=0.3, label='DNN prediction', ax=ax)
plt.xlabel('prediction')
plt.ylabel('home-care duration next week')
plt.show()
