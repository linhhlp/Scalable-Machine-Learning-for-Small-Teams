"""Web Endoint service for prediction."""

from pickle import load

import flask
import mlflow.sklearn
import pandas as pd

# Configure and Load saved model + data
MODEL_PATH = "models/sale_forecasting_sklearn_v1"
DP_PATH = "models/dp_v1.pkl"
Y_COLUMN_PATH = "models/y_columns_v1.pkl"

model = mlflow.sklearn.load_model(MODEL_PATH)
dp = load(open(DP_PATH, "rb"))
y_column = load(open(Y_COLUMN_PATH, "rb"))

# Start web framework
app = flask.Flask(__name__)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Run prediction by saved model."""
    global model, dp, y_column
    # If model is not cached anymore, we can reload it
    if not model or not dp or len(y_column):
        model = mlflow.sklearn.load_model(MODEL_PATH)
        dp = load(open(DP_PATH, "rb"))
        y_column = load(open(Y_COLUMN_PATH, "rb"))

    data = {"success": False}
    # Get Request
    params = flask.request.get_json(silent=True)
    if params is None:
        params = flask.request.args
    # If range of date is given, do the prediction
    if "range" in params.keys():
        # Creating Testing Features
        X_pred = dp.out_of_sample(steps=int(params["range"]))
        X_pred["NewYear"] = X_pred.index.dayofyear == 1
        X_pred.index.name = "date"
        y = pd.DataFrame(
            model.predict(X_pred), index=X_pred.index, columns=y_column
        )
        data["prediction"] = (
            y.stack(["store_nbr", "family"]).reset_index().to_json()
        )
        data["success"] = True

    return flask.jsonify(data)


# Because we use `gunicorn` to run the server, we comment the following line
# app.run(host='0.0.0.0', port = 8080) # if run Flask only
