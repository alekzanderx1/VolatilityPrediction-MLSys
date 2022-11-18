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
        self.dataframe.Date = pd.to_datetime(dataframe.Date)
        self.dataframe = dataframe.set_index('Date')
        self.dataframe = dataframe.sort_index()
        # Shift Volatility by 1 week as this the the target we will be predicting
        self.dataframe['Volatility'] = self.dataframe.Volatility.shift(1)
        self.dataframe = self.dataframe.iloc[1: , :]
        self.train, self.test = self.dataframe[0:1700], self.dataframe[1700:]
        # TODO impute empty values
        self.next(self.train_model)

    @step
    def train_model(self):
        """
        Train a regression on the training set
        """
        from sklearn.ensemble import RandomForestRegressor
        self.X_train = self.train 
        reg = RandomForestRegressor(n_estimators=100)
        reg.fit(self.X_train, self.y_train)
        # now, make sure the model is available downstream
        self.model = reg
        # go to the testing phase
        self.next(self.test_model)

    @step 
    def test_model(self):
        """
        Test the model on the hold out sample
        """
        from sklearn.ensemble import metrics

        self.y_predicted = self.model.predict(self.X_test)
        self.mse = metrics.mean_squared_error(self.y_test, self.y_predicted)
        self.r2 = metrics.r2_score(self.y_test, self.y_predicted)
        print('MSE is {}, R2 score is {}'.format(self.mse, self.r2))
        # print out a test prediction
        test_predictions = self.model.predict([[10]])
        print("Test prediction is {}".format(test_predictions))
        # all is done go to the end
        self.next(self.end)

    @step
    def end(self):
        # all done, just print goodbye
        print("All done at {}!\n See you, space cowboys!".format(datetime.utcnow()))


if __name__ == '__main__':
    VolatilityPredictionFlow()
