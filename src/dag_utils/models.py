import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from numpy import asarray
from sklearn.linear_model import ElasticNet

# Fits a RandomForestRegressor model and returns a prediction
def random_forest_forecast(X_train: list(), y_train: list(), testX, n_trees: int) -> float:
    # transform list into array
    trainX = asarray(X_train)
    trainy = asarray(y_train)

    # fit model
    model = RandomForestRegressor(n_estimators=n_trees)
    model.fit(trainX, trainy)
    
    # make a one-step prediction
    yhat = model.predict([testX])
    return yhat[0]

# Fits a RandomForestRegressor model and returns the fitted model
def fit_random_forest_classifier(X_train: list(), y_train: list(), n_trees: int) -> RandomForestRegressor:
    # transform list into array
    trainX = asarray(X_train)
    trainy = asarray(y_train)

    # fit model
    model = RandomForestRegressor(n_estimators = n_trees)
    model.fit(trainX, trainy)
    
    return model 

# Fits a ElasticNet model and returns a prediction
def elasticnet_forecast(X_train: list(), y_train: list(), testX, alpha: float) -> float:
    # transform list into array
    trainX = asarray(X_train)
    trainy = asarray(y_train)
    
    # fit model
    model = ElasticNet(alpha = alpha)
    model.fit(trainX, trainy)

    # make a one-step prediction
    yhat = model.predict([testX])
    return yhat[0]

# Fits a ElasticNet model and returns the fitted model
def fit_elasticnet_classifier(X_train: list(), y_train: list(), alpha: float) -> ElasticNet:
    # transform list into array
    trainX = asarray(X_train)
    trainy = asarray(y_train)
    
    # fit model
    model = ElasticNet(alpha = alpha)
    model.fit(trainX, trainy)
    
    return model
