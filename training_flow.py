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
        default='csv_dataset.csv')

    TEST_SPLIT = Parameter(
        name='test_split',
        help='Determining the split of the dataset for testing',
        default=0.20
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
        from io import StringIO
        import pandas as pd
        
        self.dataframe = pd.read_csv(StringIO(self.DATA_FILE))
        
        print("Total of {} rows in the dataset!".format(len(self.dataframe)))
        # go to the next step
        self.next(self.check_dataset)

    @step
    def check_dataset(self):
        """
        Check data is ok before training starts
        """
        #assert(all(y < 100 and y > -100 for y in self.Ys))
        assert(True)
        self.next(self.prepare_train_and_test_dataset)

    @step
    def prepare_train_and_test_dataset(self):
        # Set Date as Index
        self.dataframe.Date = pd.to_datetime(dataframe.Date)
        self.dataframe = dataframe.set_index('Date')
        self.dataframe = dataframe.sort_index()
        # Shift Volatility forward by 1 week as this the the target we will be predicting
        self.dataframe['Volatility'] = self.dataframe.Volatility.shift(1)
        self.dataframe = self.dataframe.iloc[1: , :]
        # Move target column to end
        volatility_column = self.dataframe.pop('Volatility')
        self.dataframe.insert(len(self.dataframe.columns),"Volatility",volatility_column)
        # TODO impute NaN values
        # TODO convert Mcap to float
        
        # spilit into train and test sets
        self.train, self.test = self.dataframe[0:1600], self.dataframe[1600:]
        self.next(self.train_walk_forward_validation)
        

    @step
    def train_walk_forward_validation(self):
        """
        Train a regression on the training set
        """
        from sklearn.ensemble import RandomForestRegressor
        from numpy import asarray
        
        def random_forest_forecast(train, testX):
            # transform list into array
            train = asarray(train)
            # split into input and output columns
            trainX, trainy = train[:, :-1], train[:, -1]
            # fit model
            model = RandomForestRegressor(n_estimators=10)
            model.fit(trainX, trainy)
            # make a one-step prediction
            yhat = model.predict([testX])
            return yhat[0]
        
        predictions = list()
        history = [x for x in self.train.values]
        
        for i in range(len(self.test)):
            testX, testy = self.test.iloc[i].values[:-1], self.test.iloc[i].values[-1]
            # fit model on history and make a prediction
            yhat = random_forest_forecast(history, testX)
            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop
            history.append(test.iloc[i].values)
            # summarize progress
            print('>expected=%.1f, predicted=%.1f' % (testy, yhat))            
        
        self.y_predicted = predictions
        # go to the testing phase
        self.next(self.evaluate_results)

    @step 
    def evaluate_results(self):
        """
        Calculate resulting metrics from predictions 
        """
        from sklearn.ensemble import metrics

        self.r2 = metrics.r2_score(self.test['Volatility'], self.y_predicted)
        print('R2 score is {}'.format(self.r2))
        # all is done go to the end
        self.next(self.end)

    @step
    def end(self):
        # all done, just print goodbye
        print("All done at {}!\n See you, space cowboys!".format(datetime.utcnow()))


if __name__ == '__main__':
    VolatilityPredictionFlow()
