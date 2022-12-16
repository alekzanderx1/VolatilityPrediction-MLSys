"""
    This script runs a small Flask app that displays a web page showing Volatility and Market Return Prediction
    Results and allows user to make predictions using the trained models using a HTML Form

    App also provides two endpoint GET /results and POST /predictions 
    through which user can bypass the UI entirely if required

"""

from flask import Flask, render_template, request, jsonify, Response
import numpy as np
from metaflow import Flow, get_metadata, metadata
import uuid
from time import time 

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
vol_best_param = latest_run.data.vol_best_param
er_best_r2 = latest_run.data.er_best_r2
er_best_model_type = latest_run.data.er_best_model_type
er_best_model = latest_run.data.er_best_model
er_best_param = latest_run.data.er_best_param


# We need to initialise the Flask object to run the flask app 
# By assigning parameters as static folder name,templates folder name
app = Flask(__name__, static_folder='static', template_folder='templates')

# Homepage endpoint
@app.route('/',methods=['GET'])
def main():
  # on GET we display the homepage  
  if request.method=='GET':
    return render_template('index.html', 
      project=PROJECT_NAME, 
      vol_best_r2=vol_best_r2, 
      vol_best_model_type=vol_best_model_type,
      vol_best_param=vol_best_param,
      er_best_r2=er_best_r2,
      er_best_model_type=er_best_model_type,
      er_best_param=er_best_param
      )
  else:
    return Response(
        "Only GET Method allowed",
        status=400,
    )

# Results endpoint
@app.route('/results',methods=['GET'])
def results():
  # on GET we return model traning results and summary  
  if request.method=='GET':
    start = time()
    # Initialize variables 
    eventId = str(uuid.uuid4())
    result = dict()
    result['data'] = dict()
    result['metadata'] = dict()
    
    # Debug
    print(f" EventId: {eventId}, Request Type: GET /results")

    # Prepare results
    result['data']['vol_best_r2'] = vol_best_r2
    result['data']['vol_best_model_type'] = vol_best_model_type
    result['data']['vol_best_param'] = vol_best_param
    result['data']['er_best_r2'] = er_best_r2
    result['data']['er_best_model_type'] = er_best_model_type
    result['data']['er_best_param'] = er_best_param
    result['metadata']['eventId'] = eventId
    result['metadata']['serverTimestamp'] = int(time())
    end = time()
    result['metadata']['time'] = end - start

    return jsonify(result)
  else:
    return Response(
        "Only GET Method allowed",
        status=400,
    )

# Predictions endpoint
@app.route('/predict',methods=['POST'])
def predict():
  # on POST we make a prediction over the input text supplied by the user
  if request.method=='POST':
    start = time()
    # Initialize variables 
    eventId = str(uuid.uuid4())
    result = dict()
    result['data'] = dict()
    result['metadata'] = dict()

    # Read request parameter
    request_data = request.get_json()

    print(f"Request_data: {request_data}")

    # read values
    pipelineType = request_data['pipelineType']
    spread = float(request_data['spread'])
    mcap = float(request_data['mcap'])
    pe = float(request_data['pe'])
    pb = float(request_data['pb'])
    close = float(request_data['close'])
    yeild30 = float(request_data['yeild30'])
    aayeild = float(request_data['aayeild'])
    smb = float(request_data['smb'])
    hml = float(request_data['hml'])
    rf = float(request_data['rf'])
    yield_val = float(request_data['yield'])
    m3 = float(request_data['m3'])
    inflation = float(request_data['inflation'])
    lagValues = request_data['lag']

    if pipelineType == 'Volatility':
      mkt_rf = float(request_data['mkt_rf'])
    else:
      volatility = float(request_data['volatility'])

    # Create Input for algorithm

    # Perform prediction using latest model
    if pipelineType == 'Volatility':
      input1 = [spread,mcap,pb,pe,close,yeild30,aayeild,mkt_rf,smb,hml,rf,yield_val,m3,inflation]
      input2 = [float(x) for x in lagValues.split(',')]
      xTest = input1 + input2
      val = vol_best_model.predict([xTest])
    else:
      input1 = [spread,mcap,pb,pe,close,volatility, yeild30,aayeild,smb,hml,rf,yield_val,m3,inflation]
      input2 = float(lagValues)
      input1.append(input2)
      val = er_best_model.predict([input1])
    
    #  debug
    print(f" EventId: {eventId}, prediction: {val[0]}")
    
    # Contruct Response
    result['data']['prediction'] = val[0]
    result['metadata']['eventId'] = eventId
    result['metadata']['serverTimestamp'] = int(time())
    end = time()
    result['metadata']['time'] = end - start
    
    # Returning the response to the client as JSON
    return jsonify(result)
  else:
    return Response(
        "Only POST Method allowed",
        status=400,
    )
    

if __name__=='__main__':
  # Run the Flask app to run the server
  app.run(debug=True)