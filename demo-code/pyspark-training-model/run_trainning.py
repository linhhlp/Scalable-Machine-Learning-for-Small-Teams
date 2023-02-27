from pyspark.context import SparkContext
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
from pyspark.sql.session import SparkSession
import time

# Define a function that collects the features of interest
# (date, store_nbr, and family) into a vector.
# Package the vector in a tuple containing the label (`sales`) for that row.
def vector_from_inputs(r):
      return (float(r["sales"]), Vectors.dense(time.mktime(r["date"].timetuple()),
                                            float(r["store_nbr"]),))
sc = SparkContext()
spark = SparkSession(sc)

# Read the data from BigQuery as a Spark Dataframe.
sales_data = spark.read.format("bigquery").option('project','scalable-model-piplines').option(
    "table", "store_sales.simplified_data_table").load()
# Create a view so that Spark SQL queries can be run against the data.
sales_data.createOrReplaceTempView("sales_data")

query = """
    SELECT date, store_nbr, family, sales
    FROM `sales_data` 
"""
clean_data = spark.sql(query)
# Create an input DataFrame for Spark ML using the above function.
training_data = clean_data.rdd.map(vector_from_inputs).toDF(["label",
                                                             "features"])
training_data.cache()
# Construct a new LinearRegression object and fit the training data.
lr = LinearRegression(maxIter=5, regParam=0.2, solver="normal")
model = lr.fit(training_data)
# Print the model summary.
print("Coefficients:" + str(model.coefficients))
print("Intercept:" + str(model.intercept))
print("R^2:" + str(model.summary.r2))
model.summary.residuals.show()