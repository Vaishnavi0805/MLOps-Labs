# Import necessary libraries and modules
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.lab import load_data, data_preprocessing, build_save_model, load_model_elbow

# Define default arguments for the DAG
default_args = {
    'owner': 'Vaishnavi Sarmalkar',
    'start_date': datetime(2026, 3, 15),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create a DAG instance
dag = DAG(
    'mall_customer_clustering_dag',
    default_args=default_args,
    description='Customer Segmentation using K-Means Clustering on Mall Customers Dataset',
    schedule_interval=None,  # Manual triggering
    catchup=False,
)

# Task 1: Load data from CSV file
load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag,
)

# Task 2: Preprocess the data (normalize features)
data_preprocessing_task = PythonOperator(
    task_id='data_preprocessing_task',
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
    dag=dag,
)

# Task 3: Build and save KMeans clustering model
build_save_model_task = PythonOperator(
    task_id='build_save_model_task',
    python_callable=build_save_model,
    op_args=[data_preprocessing_task.output, "customer_clustering_model.pkl"],
    dag=dag,
)

# Task 4: Load model and determine optimal clusters using elbow method
load_model_task = PythonOperator(
    task_id='load_model_elbow_task',
    python_callable=load_model_elbow,
    op_args=["customer_clustering_model.pkl", build_save_model_task.output],
    dag=dag,
)

# Set task dependencies
load_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task

# Allow command-line interaction
if __name__ == "__main__":
    dag.cli()