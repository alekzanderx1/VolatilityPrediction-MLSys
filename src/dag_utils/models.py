import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from numpy import asarray

def random_forest_forecast(train, testX, params):
    # transform list into array
    train = asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = RandomForestRegressor(n_estimators=30)
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict([testX])
    return yhat[0]