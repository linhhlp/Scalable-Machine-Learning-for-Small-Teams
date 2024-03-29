"""Demo how to run the Model Trainer."""

import os
from datetime import date
from pickle import dump

import gcsfs
import pandas as pd
from google.cloud import bigquery
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "newacc_gcp_credential.json"


def upload_file_to_cloud_storage(src_dir: str, gcs_dst: str, recursive=False):
    """Upload a directory of files to Google Cloud Storage."""
    fs = gcsfs.GCSFileSystem()
    fs.put(src_dir, gcs_dst, recursive=recursive)


def run_Model_Trainer():
    """Run Model Trainer."""
    client = bigquery.Client()

    query = """
        SELECT date, store_nbr, family, sales
        FROM `scalable-model-piplines.store_sales.simplified_data_table` 
    """

    train_data = client.query(query).to_dataframe()
    train_data["date"] = pd.to_datetime(train_data.date).dt.to_period("D")
    train_data = train_data.set_index(
        ["store_nbr", "family", "date"]
    ).sort_index()

    # Training Range
    range_begin = "2016-08-01"
    range_end = "2017-08-01"
    # Create Labels data (y output)
    y = (
        train_data.unstack(["store_nbr", "family"])
        .loc[range_begin:range_end]
        .fillna(0)
    )

    # Feature Engineering
    # Support Seasionality
    fourier = CalendarFourier(freq="M", order=30)
    dp = DeterministicProcess(
        index=y.index,
        order=1,
        seasonal=True,
        additional_terms=[fourier],
        period=7,
    )
    X = dp.in_sample()
    X["NewYear"] = X.index.dayofyear == 1

    # Modeling Fitting
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    path = "tmp/"
    model_name = "sale_forecasting_sklearn"
    dp_name = "dp"
    y_columns = "y_columns"
    today = str(date.today())

    # Save the model
    import mlflow.sklearn

    mlflow.sklearn.save_model(model, path + model_name + today)
    dump(y.columns, open(path + y_columns + today + ".pkl", "wb"))

    # Save the Seasonality (or DeterministicProcess)
    # save the scaler
    dump(dp, open(path + dp_name + today + ".pkl", "wb"))

    # Upload to Google Cloud Storage

    bucket_name = "gcs://scalable-model-piplines-trained_model/"

    upload_file_to_cloud_storage(
        path + model_name + today,
        bucket_name + model_name + today,
        recursive=True,
    )

    upload_file_to_cloud_storage(
        path + dp_name + today + ".pkl", bucket_name + dp_name + today + ".pkl"
    )

    upload_file_to_cloud_storage(
        path + y_columns + today + ".pkl",
        bucket_name + y_columns + today + ".pkl",
    )

    # Clean up
    import shutil

    shutil.rmtree(path)
