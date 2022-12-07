"""

MAKE SURE TO RUN THIS WITH METAFLOW LOCAL FIRST

"""


from metaflow import FlowSpec, step, Parameter, IncludeFile, current
from datetime import datetime
import os


# make sure we are running locally for this
assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_ENVIRONMENT', 'local') == 'local'


class VolatilityPredictionFlow(FlowSpec):
    """
    VolatilityPredictionFlow is a DAG reading data from a file 
    and training a Regression model successfully.
    """
    
    # if a static file is part of the flow, 
    # it can be called in any downstream process,
    # gets versioned etc.
    # https://docs.metaflow.org/metaflow/data#data-in-local-files
    DATA_FILE = IncludeFile(
        'dataset',
        help='Text file with the dataset',
        is_text=True,
        default='final.csv')

    TEST_SPLIT = Parameter(
        name='test_split',
        help='Determining the split of the dataset for testing',
        default=0.30
    )

    @step
    def start(self):
        """
        Start up and print out some info to make sure everything is ok metaflow-side
        """
        print("Starting up at {}".format(datetime.utcnow()))
        # debug printing - this is from https://docs.metaflow.org/metaflow/tagging
        # to show how information about the current run can be accessed programmatically
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)
        self.next(self.load_data)

    @step
    def load_data(self): 
        """
        Read the data in from the static file
        """
        import pandas as pd
        from io import StringIO
        
        self.df = pd.read_csv(StringIO(self.DATA_FILE))
        self.df.set_index('Date', inplace = True)
        self.df = self.df.sort_index()
        self.next(self.clean_transform_dataset)
    
    @step
    def clean_transform_dataset(self) -> None:
        """
        Manipulate the dataframe to fill missing rows and split in two seperate dataframes
        """
        def value_to_float(x):
            if type(x) == float or type(x) == int:
                return x
            if 'M' in x:
                if len(x) > 1:
                    return float(x.replace('M', '')) * 1000000
                return 1000000.0
            return 0.0

        # Fix datatype for Mcap column
        self.df['Mcap'] = self.df['Mcap'].apply(value_to_float) 

        # TODO: Normalize all values

        #creating two different dataframes for volatility and excess returns
        self.vol_df = self.df.copy()
        self.ret_df = self.df.copy()
        
        #Removing all the dates where we don't have volatility value
        self.vol_df.dropna(subset = ['Volatility'], inplace = True)

        #adding lagged variables based on exploratory data analysis
        lagged_data = []
        for i in range(1,10):
            col_label = "Volshift" + str(i)
            self.vol_df[col_label] = self.vol_df['Volatility'].shift(i)
        
        #forward fill all the rows
        self.vol_df.ffill(inplace = True)

        #drop all the rows with NA
        self.vol_df.dropna(inplace = True)

        #Removing all the dates where we don't have excess return value
        self.ret_df.dropna(subset = ['Mkt_rf'], inplace = True)

        #adding lagged variables based on 
        self.ret_df['Mkt_rf_shifted'] = self.ret_df['Mkt_rf'].shift(1)

        #forward fill all the rows
        self.ret_df.ffill(inplace = True)

        #drop all the rows with NA
        self.ret_df.dropna(inplace = True)
        
        
        # Shift both Volatility and Excess Return back by 1 week since our target is the value for next week
        self.vol_df['Volatility'] = self.vol_df.Volatility.shift(-1)
        self.ret_df['Mkt_rf'] = self.vol_df.Mkt_rf.shift(-1)
        self.vol_df.dropna(inplace = True)
        self.ret_df.dropna(inplace = True)

        # split the dataframes in X and y dataframes
        vol_col = [x for x in self.vol_df.keys() if x != 'Volatility']
        self.vol_df_X = self.vol_df[vol_col]
        self.vol_df_y = self.vol_df['Volatility']

        ret_col = [x for x in self.ret_df.keys() if x != 'Mkt_rf']
        self.ret_df_X = self.ret_df[ret_col]
        self.ret_df_y = self.ret_df['Mkt_rf']

        self.next(self.check_dataset)
        
    @step
    def check_dataset(self):
        """
        Check data is ok before training starts
        """
        #assert(all(y < 100 and y > -100 for y in self.Ys))
        # TODO Implement checks on Data Later
        assert(True)
        self.next(self.train_test_split)

    @step
    def train_test_split(self):
        import pandas as pd
        
        len_train_vol = round(len(self.vol_df)*0.7)
        len_train_ret = round(len(self.ret_df)*0.7)
        
        # Train Test split for Volatility dataset
        self.vol_train_X, self.vol_test_X = self.vol_df_X[:len_train_vol], self.vol_df_X[len_train_vol:]
        self.vol_train_y, self.vol_test_y = self.vol_df_y[:len_train_vol], self.vol_df_y[len_train_vol:]
        
        # Train Test split for Market Return dataset
        self.ret_train_X, self.ret_test_X = self.ret_df_X[:len_train_ret], self.ret_df_X[len_train_ret:]
        self.ret_train_y, self.ret_test_y = self.ret_df_y[:len_train_ret], self.ret_df_y[len_train_ret:]

        self.pipeline_types = ['VolatilityPrediction','ExcessReturnPrediction']
        self.next(self.begin_prediction_pipeline, foreach="pipeline_types")

    @step
    def begin_prediction_pipeline(self):
        self.pipeline_type = self.input
        print(f'Beginning {self.pipeline_type} Pipeline')

        # Choose Train and Test dataset to pass on based on choice of Pipeline
        if self.pipeline_type == 'VolatilityPrediction':
            self.X_train = self.vol_train_X
            self.y_train = self.vol_train_y
            self.X_test = self.vol_test_X
            self.y_test = self.vol_test_y
        else:
            self.X_train = self.ret_train_X
            self.y_train = self.ret_train_y
            self.X_test = self.ret_test_X
            self.y_test = self.ret_test_y 

        self.classifier_types = ['RandomForest','ElasticNet']
        self.next(self.train_with_walk_forward_validation, foreach="classifier_types")

    @step
    def train_with_walk_forward_validation(self):
        """
        Train a Regression model on train set and predict on test set in a walk forward fashion
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import ElasticNet
        from numpy import asarray
        
        self.classifier_type = self.input
                
        def random_forest_forecast(X_train, y_train, testX):
            # transform list into array
            trainX = asarray(X_train)
            trainy = asarray(y_train)

            # fit model
            model = RandomForestRegressor(n_estimators=30)
            model.fit(trainX, trainy)
            
            # make a one-step prediction
            yhat = model.predict([testX])
            return yhat[0]

        def elasticnet_forecast(X_train, y_train, testX):
            # transform list into array
            trainX = asarray(X_train)
            trainy = asarray(y_train)
            
            # fit model
            model = ElasticNet()
            model.fit(trainX, trainy)

            # make a one-step prediction
            yhat = model.predict([testX])
            return yhat[0]
        
        predictions = list()
        history_X = [x for x in self.X_train.values]
        history_Y = [y for y in self.y_train.values]
        
        for i in range(len(self.X_test)):
            testX = self.X_test.iloc[i].values
            testY = self.y_test.iloc[i]
            # fit model on history and make a prediction
            if self.classifier_type == 'RandomForest':
                yhat = random_forest_forecast(history_X, history_Y, testX)
            else:
                yhat = elasticnet_forecast(history_X, history_Y, testX)

            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop
            history_X.append(self.X_test.iloc[i].values)     
            history_Y.append(self.y_test.iloc[i])    
        
        self.y_predicted = predictions
        # go to the evaluation phase
        self.next(self.evaluate_classifier)  

    @step
    def evaluate_classifier(self):
        """
        Calculate resulting metrics from predictions for given classifier
        """
        from sklearn import metrics

        self.r2 = metrics.r2_score(self.y_test, self.y_predicted)
        print('R2 score is {}'.format(self.r2))

        self.next(self.evaluate_pipeline)

    @step
    def evaluate_pipeline(self, inputs):
        # combine results from both algorithms 
        # print and store results and best model/params
        self.merge_artifacts(inputs, exclude=['y_predicted','classifier_type','r2'])
        for clf in inputs:
            print(f" {clf.classifier_type} Classifier's R2 score {clf.r2} for {self.pipeline_type} Pipeline")

        self.next(self.combine_pipeline_results)


    @step
    def combine_pipeline_results(self, inputs):
        # Store results and best model in artifacts to use in Flask app
        print('Pipelines joined!')
        self.next(self.end)


    @step
    def end(self):
        # all done, just print goodbye
        print("All done at {}!\n See you, space cowboys!".format(datetime.utcnow()))


if __name__ == '__main__':
    VolatilityPredictionFlow()
