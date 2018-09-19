# coding:utf-8
# Created by chen on 18/09/2018
# email: q.chen@student.utwente.nl

##DNN performance on some aspect of the data

from tools import *
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import numpy as np
''''
train data 
'''
## load data
def getPerformance(weekNum,col_names):
    '''
    :param weekNum: how many weeks after nurse assessment
    :param col_names: could be  Marital_status, Living_unit, JobTitle, estimated_care_request, estimated_care_duration,
    TeamID, ProblemName
    :return: save performance on specified categorical data
    '''
    pandas2ri.activate()
    readRDS = robjects.r['readRDS']
    fileName = "../../data/allCom_v" + str(weekNum)+"_no_scale.rds"
    df_all = readRDS(fileName)
    df_all = pandas2ri.ri2py(df_all)

    fileName ="../../data/allCom_v"+str(weekNum)+"_final.rds"
    df = readRDS(fileName)
    df = pandas2ri.ri2py(df)
    # df = pd.read_csv(fileName)
    df = df.rename({'duration_next_week':'label'}, axis='columns')

    filter_col = [col for col in df_all if col.startswith(col_names)]
    removed_col = [col for col in df if col.startswith(col_names)]
    filter_col = list(set(filter_col)^set(removed_col))

    df_filtered = df_all[filter_col]

    df_filtered.reset_index(drop=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df= pd.concat([df,df_filtered], axis=1)

    results = {}
    for filter_colName in filter_col:
        for filters, filter_df in df.groupby([filter_colName], axis=0):
            # scaled feature,so 0 and 1 boolean value changed into >0 and <0 two categories
            if ((filter_df[filter_colName] > 0).all()):
                filter_df = filter_df.drop(filter_col, axis=1)
                preds = filter_df['label']
                filter_df = filter_df.drop('label', axis=1)
                X = filter_df.values
                y = preds.values
                results_temp = {}
                results_temp['colName'] = filter_colName
                for j in range(5):
                    name_weights = "model/v"+str(weekNum)+"/final_model_fold" + str(j) + "_weights.h5"
                    model2 = load_model(name_weights,custom_objects={'rmse':rmse, 'r_square':r_square})
                    y_pred = model2.predict(X)
                    MAE_col = "MAE_" +filter_colName
                    RMSE_col = "RMSE_" + filter_colName
                    R2_col = "R2"+filter_colName
                    results_temp.setdefault(MAE_col, []).append(sklearn.metrics.mean_absolute_error(y, y_pred))
                    results_temp.setdefault(RMSE_col, []).append(math.sqrt(sklearn.metrics.mean_squared_error(y, y_pred)))
                    results_temp.setdefault(R2_col, []).append(sklearn.metrics.r2_score(y, y_pred))

                results_temp[MAE_col+"_average"] = [sum(results_temp[MAE_col])/len(results_temp[MAE_col])]
                results_temp[RMSE_col+"_avergae"] = [sum(results_temp[RMSE_col]) / len(results_temp[RMSE_col])]
                results_temp[R2_col+"_average"] = [sum(results_temp[R2_col]) / len(results_temp[R2_col])]
                # calculate variance using a list comprehension
                results_temp[MAE_col+"_variance"] = np.var(results_temp[MAE_col])
                results_temp[RMSE_col + "_variance"] = np.var(results_temp[RMSE_col])
                results_temp[R2_col+"_variance"] = np.var(results_temp[R2_col])
                # if results is empty dict then bool(results) returns false
                if bool(results):
                    results.update(results_temp)
                else:
                    results = results_temp.copy()

    ## write results into txt files
    fileName = "results/performance_"+str(col_names)+"_week_"+str(weekNum)+".txt"
    with open(fileName, "w") as file:
        for key, value in results.items():
            file.write("%s:%s\n" % (key, value))

# calcualte performance on one week after nurse assessment on "estimated_care_request" category
getPerformance(1,'estimated_care_duration')