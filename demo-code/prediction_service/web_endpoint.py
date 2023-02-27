import pandas as pd
import mlflow.sklearn
import flask
import json
from pickle import load

#### Configure and Load saved model + data ########
from model_trainer import run_Model_Trainer

############ Start web framework ###################
app = flask.Flask(__name__)

@app.route("/run", methods=["GET","POST"])
def predict():
    ######## If model is not cached anymore, we can reload it ###########
    run_Model_Trainer()
        
    data = {"success": True}

    return flask.jsonify(data)

#app.run(host='0.0.0.0', port = 8080) # if run Flask only