"""This serves as Web Endpoints for Model Trainer."""

import flask

# Configure and Load saved model + data
from model_trainer import run_Model_Trainer

# Start web framework
app = flask.Flask(__name__)


@app.route("/run", methods=["GET", "POST"])
def predict():
    """Run Model Trainer."""
    # If model is not cached anymore, we can reload it
    run_Model_Trainer()

    data = {"success": True}

    return flask.jsonify(data)


# Commented out because we use `gunicorn` to run the app as a server
# app.run(host='0.0.0.0', port = 8080) # if run Flask only
