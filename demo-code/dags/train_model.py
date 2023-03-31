from datetime import datetime, timedelta

from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
from tasks import task1_run_Model_Trainer, task2_upload_Model

default_dag_args = {}

with DAG(
    dag_id="train_model",
    default_args=default_dag_args,
    start_date=datetime(2023, 2, 25, 0, 0),
    # schedule_interval=timedelta(days=1), # every day
    # At 21:15 every day
    schedule_interval="15 21 * * *",
) as dag:
    do_stuff1 = PythonOperator(
        task_id="task_1",
        python_callable=task1_run_Model_Trainer.main,  # entrypoint is main()
    )
    do_stuff2 = PythonOperator(
        task_id="task_2",
        python_callable=task2_upload_Model.main,  # assume entrypoint is main()
    )
    do_stuff1 >> do_stuff2
