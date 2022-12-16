# VolatilityAndMarketReturnPrediction

Repository containing code and documentation for Final project for Machine Learning in Financial Engineering course at NYU.

Goal is to build two kind of regression models, one for Volatility Prediction and one for Market Excess Return prediction. These in turn can be used to perform portfolio allocation.

The models are trained, deployed, and served on the cloud using AWS EC2. 

## Data Description

Data manually obtained through Bloomberg. 

Features are based on paper titled *Machine learning portfolio allocation* by *Michael Pinelis* and *David Ruppert*.   


## Project Structure

* `src` is the folder containing the scripts for Metaflow and Flask apps
* `src/dag_utils` is a custom package built to be used in metaflow pipeline
* `Data` folder contains all the data, cleaned or uncleaned used during various stages of the project
* `notebooks` folder contains Jupyter notebooks used to perform EDA and experimental Model Training
* `requirements.txt` file contains all Python libraries required to run the project, can be installed using pip

## Intructions to run pipeline locally

* Install requirements using `pip install -r requirements.txt` and then goto `src` folder using `cd src`
* Run metaflow dag using `python training_flow.py run` 
* Confirm location of Metaflow artifacts from logs, and update it in `app.py` line 18.
* Launch Flask App using `python app.py`

> Note: Running step 2 i.e. training_flow.py can take 20-30 minutes to finish execution due to large amount of models being trained

The flask app should be available at http://127.0.0.1:5000/

## Instructions for deploying project on AWS

Follow instructions as given [here](https://www.twilio.com/blog/deploy-flask-python-app-aws) to setup an EC2 instance. Follow the instuctions until *Transfer your project files to remote host step*

Then follow below steps to launch the application:

* Follow first three steps from [above section](#intructions-to-run-pipeline-locally)
* Launch tmux session using `tmux new -s FlaskApp` 
* Launch Flask app using `FLASK_APP=app.py flask run --host=0.0.0.0 --port=8080`
* Press Ctrl B and press D on your keyboard to leave the tmux session running in the background.
* To stop the session or make changes you can login to Tmux session again using  `tmux attach -t FlaskApp`

While the app is running, it should be available at Public IP of your EC2 instance at port 8080.



