"""
    This script runs a small Flask app that displays a simple web form for users to see Classification Model results.

"""

from flask import Flask, render_template, request
import numpy as np
from metaflow import Flow
from metaflow import get_metadata, metadata
from flask import jsonify

#### THIS IS GLOBAL, SO OBJECTS LIKE THE MODEL CAN BE RE-USED ACROSS REQUESTS ####

PROJECT_NAME = 'MLSys-FinalProject'
FLOW_NAME = 'VolatilityAndExcessReturnPredictionFlow' # name of the target class that generated the model
# Set the metadata provider as the src folder in the project,
# which should contains /.metaflow
metadata('/home/syed/')
# Fetch currently configured metadata provider to check it's local!
print(get_metadata())

def get_latest_successful_run(flow_name: str):
    "Gets the latest successfull run."
    for r in Flow(flow_name).runs():
        if r.successful: 
            return r

# get artifacts from latest run, using Metaflow Client API
latest_run = get_latest_successful_run(FLOW_NAME)
vol_best_r2 = latest_run.data.vol_best_r2
vol_best_model_type = latest_run.data.vol_best_model_type
vol_best_model = latest_run.data.vol_best_model
er_best_r2 = latest_run.data.er_best_r2
er_best_model_type = latest_run.data.er_best_model_type
er_best_model = latest_run.data.er_best_model


# We need to initialise the Flask object to run the flask app 
# By assigning parameters as static folder name,templates folder name
app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/',methods=['GET'])
def main():
  # on GET we display the homepage  
  if request.method=='GET':
    return render_template('index.html', 
      project=PROJECT_NAME, 
      vol_best_r2=vol_best_r2, 
      vol_best_model_type=vol_best_model_type,
      er_best_r2=er_best_r2,
      er_best_model_type=er_best_model_type
      )


@app.route('/predict',methods=['GET'])
def predict():
  return jsonify({"hello":"hello"})

    # # on POST we serve model test results to fronend to display
  # if request.method=='POST':
  #   request_data = request.get_json()
  #   gender = request_data['gender']
  #   #  debug
  #   print(gender)
  #   # Returning the response to the client
  #   missRateResponse = overAllMissRate
  #   if(gender in ['Male','Female','Other']):
  #       missRateResponse = genderStats[gender]
  #   return jsonify({"gender":gender, "testMissRate":overAllMissRate,"groupMissRate":missRateResponse})
    

if __name__=='__main__':
  # Run the Flask app to run the server
  app.run(debug=True)