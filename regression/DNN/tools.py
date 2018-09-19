# coding:utf-8
# Created by chen on 18/09/2018
# email: q.chen@student.utwente.nl

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from keras.layers import *
from keras.models import *
from keras.callbacks import *
import matplotlib.pyplot as plt
from keras.models import load_model
import keras.losses
import sklearn.metrics, math

def rmse(y_true, y_pred):
    from keras import backend
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# mean squared error (mse) for regression
def mse(y_true, y_pred):
    from keras import backend
    return backend.mean(backend.square(y_pred - y_true), axis=-1)

# coefficient of determination (R^2) for regression
def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

def r_square_loss(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ( 1 - SS_res/(SS_tot + K.epsilon()))

def load_data_kfold(k,X,Y):
    ''''
    cross validation
    '''
    folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=1).split(X,Y))
    return folds

def get_model(X):
    ''''
     define model, each ctaegorical data are into embedding layer
    '''
    model = Sequential()
    model.add(Dense(1000, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=1, activation='relu'))
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error", rmse, r_square])
    # model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['mae','mse','mape'])
    return model

def get_callbacks(name_weights, patience_lr):
    ''''
    early stop
    reduce learning rate whencompile loss doesnt reduce in the next patience_lr times epoches
    save best model weight
    '''
    mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')
    early_stopping = EarlyStopping(monitor="mean_squared_error", patience=40, verbose=1, mode='auto')
    return [mcp_save, reduce_lr_loss,early_stopping]

