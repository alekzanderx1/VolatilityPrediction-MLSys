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
        import pandas as pd
        
        self.df = pd.read_csv("final.csv")
        self.df.set_index('Date', inplace = True)
        
        self.next(self.check_dataset)

    @step
    def check_dataset(self):
        """
        Check data is ok before training starts
        """
        #assert(all(y < 100 and y > -100 for y in self.Ys))
        # TODO Implement checks on Data Later
        assert(True)
        self.next(self.prepare_train_and_test_dataset)
    
    @step
    def process_data(self) -> None:
        """
        Manipulate the dataframe to fill missing rows and split in two seperate dataframes
        """

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
        self.ret_df['Mkt_rf_shifted'] = self.ret_df['Mkt_df'].shift(1)

        #forward fill all the rows
        self.ret_df.ffill(inplace = True)

        #drop all the rows with NA
        self.ret_df.dropna(inplace = True)

        #split the dataframes in X and y dataframes
        vol_col = [x for x in self.vol_df.keys() if x != 'Volatility']
        self.vol_df_X = self.vol_df[vol_col]
        self.vol_df_y = self.vol_df['Volatility']

        ret_col = [x for x in self.ret_df.keys() if x != 'Mkt_rf']
        self.ret_df_X = self.ret_df[ret_col]
        self.ret_df_y = self.ret_df['Mkt_rf']

        self.next(self.train_test_split)

    @step
    def train_test_split(self):
        import pandas as pd
        
        len_train_vol = round(len(self.vol_df)*0.7)
        len_train_ret = round(len(self.ret_df)*0.7)
        
        self.vol_train_X, self.vol_test_X = self.vol_df_X[:len_train_vol], self.vol_df_X[len_train_vol:]
        self.vol_train_y, self.vol_test_y = self.vol_df_y[:len_train_vol], self.vol_df_y[len_train_vol:]
        
        self.ret_train_X, self.ret_test_X = self.ret_df_X[:len_train_ret], self.ret_df_X[len_train_ret:]
        self.ret_train_y, self.ret_test_y = self.ret_df_y[:len_train_ret], self.ret_df_y[len_train_ret:]

        self.next(self.train_walk_forward_validation)
        

    @step
    def train_walk_forward_validation(self):
        """
        Train a regression model on train set and predict on test set in a walk forward fashion
        """
        from sklearn.ensemble import RandomForestRegressor
        from numpy import asarray
        
        def random_forest_forecast(train, testX):
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
        
        predictions = list()
        history = [x for x in self.train.values]
        
        for i in range(len(self.test)):
            testX, testy = self.test.iloc[i].values[:-1], self.test.iloc[i].values[-1]
            # fit model on history and make a prediction
            yhat = random_forest_forecast(history, testX)
            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop
            history.append(self.test.iloc[i].values)
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
        from sklearn import metrics

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
