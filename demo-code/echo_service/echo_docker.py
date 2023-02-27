import pandas as pd
import mlflow.sklearn
import flask
import json
from pickle import load

#### Configure and Load saved model + data ########
model_path = "models/sale_forecasting_sklearn_v1"
dp_path = "models/dp_v1.pkl"
y_column_path = "models/y_columns_v1.pkl"

model = mlflow.sklearn.load_model(model_path)
dp = load(open(dp_path, 'rb'))
y_column = load(open(y_column_path, 'rb'))

############ Start web framework ###################
app = flask.Flask(__name__)

@app.route("/predict", methods=["GET","POST"])
def predict():
    global model, dp, y_column
    ######## If model is not cached anymore, we can reload it ###########
    if not model or not dp or len(y_column):
        model = mlflow.sklearn.load_model(model_path)
        dp = load(open(dp_path, 'rb'))
        y_column = load(open(y_column_path, 'rb'))
        
    data = {"success": False}
    ############# get Request ##################
    params = flask.request.get_json(silent=True)
    if params is None:
        params = flask.request.args
    ############### If range of date is given, do the prediction ##############
    if "range" in params.keys(): 
        X_pred = dp.out_of_sample(steps=int(params["range"])) # Creating Testing Features
        X_pred['NewYear'] = (X_pred.index.dayofyear == 1)
        X_pred.index.name = 'date'
        y = pd.DataFrame(model.predict(X_pred), index=X_pred.index, columns=y_column)
        data["prediction"] = y.stack(['store_nbr', 'family']).reset_index().to_json()
        data["success"] = True


    return flask.jsonify(data)

#app.run(host='0.0.0.0', port = 8080) # if run Flask only