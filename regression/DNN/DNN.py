# coding:utf-8
# Created by chen on 23/08/2018
# email: q.chen@student.utwente.nl
from tools import *
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
''''
train data 
'''

def trainDNN(weekNum, kFold):
    '''
    :param weekNum: how many weeks after nurse assessment
    :param kFold: kFold-cross validation
    :return: none, save model and plot
    '''
    pandas2ri.activate()
    readRDS = robjects.r['readRDS']
    fileName = "../../data/allCom_v" + str(weekNum)+"_final.rds"
    df = readRDS(fileName)
    df = pandas2ri.ri2py(df)
    df = df.rename({'duration_next_week':'label'}, axis='columns')

    preds  = df['label']
    df = df.drop('label',axis=1)
    X = df.values
    y = preds.values

    k = kFold
    folds = load_data_kfold(k,df,preds)
    # get model, and check structure
    model = get_model(X)
    print("model structure: \n", model.summary())
    batch_size = 256
    # for train_index, test_index in skf.split(X, y):
    for j, (train_idx, val_idx) in enumerate(folds):
        print('\nFold ', j)
        print(df.shape)
        X_train_cv = X[train_idx]
        y_train_cv = y[train_idx]
        X_valid_cv = X[val_idx]
        y_valid_cv = y[val_idx]
        name_weights = "model/v1/final_model_fold" + str(j) + "_weights.h5"
        callbacks = get_callbacks(name_weights=name_weights, patience_lr=10)
        model = get_model(X)
        result = model.fit(
            X_train_cv,
            y_train_cv,
            batch_size=batch_size,
            epochs=500,
            shuffle=True,
            verbose=0,
            validation_data=(X_valid_cv, y_valid_cv),
            callbacks=callbacks)
        #predict
        y_pred = model.predict(X_valid_cv)

        # -----------------------------------------------------------------------------
        # Plot learning curves including R^2 and RMSE
        # -----------------------------------------------------------------------------
        # plot training curve for R^2 (beware of scale, starts very low negative)
        plt.plot(result.history['val_r_square'])
        plt.plot(result.history['r_square'])
        plt.title('model R^2')
        plt.ylabel('R^2')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("plot/v1/r_square_" + str(j) + "_fold.png", dpi=100)
        plt.clf()
        # plot training curve for rmse
        plt.plot(result.history['rmse'])
        plt.plot(result.history['val_rmse'])
        plt.title('rmse')
        plt.ylabel('rmse')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("plot/v1/rmse_" + str(j) + "_fold.png", dpi=100)
        plt.clf()
        # print the linear regression and display datapoints
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(y_valid_cv.reshape(-1, 1), y_pred)
        y_fit = regressor.predict(y_pred)

        reg_intercept = round(regressor.intercept_[0], 4)
        reg_coef = round(regressor.coef_.flatten()[0], 4)
        reg_label = "y = " + str(reg_intercept) + "*x +" + str(reg_coef)

        plt.scatter(y_valid_cv, y_pred, color='blue', label='data')
        plt.plot(y_pred, y_fit, color='red', linewidth=2, label='Linear regression\n' + reg_label)
        plt.title('Linear Regression')
        plt.legend()
        plt.xlabel('observed')
        plt.ylabel('predicted')
        plt.savefig("plot/v1/linear_regression_" + str(j) + "_fold.png", dpi=100)
        plt.clf()
        # -----------------------------------------------------------------------------
        # print statistical figures of merit
        # -----------------------------------------------------------------------------

        import sklearn.metrics, math
        print("\n")
        print("Mean absolute error (MAE):      %f" % sklearn.metrics.mean_absolute_error(y_valid_cv, y_pred))
        print("Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(y_valid_cv, y_pred))
        print("Root mean squared error (RMSE): %f" % math.sqrt(sklearn.metrics.mean_squared_error(y_valid_cv, y_pred)))
        print("R square (R^2):                 %f" % sklearn.metrics.r2_score(y_valid_cv, y_pred))

# train data that one week after nurse assessment
# use 5-cross validation
trainDNN(1,5)






