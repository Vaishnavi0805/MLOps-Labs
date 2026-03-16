# Airflow Lab: Mall Customer Segmentation Pipeline


## Overview

This project implements an ML pipeline using Apache Airflow to segment mall customers using K-Means clustering with automated elbow method optimization.

## Project Structure
```
Airflow_Lab/
├── dags/
│   ├── airflow_clustering.py    # Airflow DAG definition
├── src/
│   └── lab.py                   # ML pipeline functions
├── data/
│   └── customers.csv            # Mall Customers dataset
├── model/                       # Saved models (generated)
│   └── customer_clustering_model.pkl
├── logs/                        # Airflow logs (generated)
├── docker-compose.yaml          # Docker configuration
├── setup.sh                     # Setup script
└── README.md
```

## Pipeline Architecture

The DAG consists of **4 tasks** executed sequentially:

1. **load_data_task**: Loads Mall Customers CSV data
2. **data_preprocessing_task**: Normalizes Annual Income & Spending Score using MinMaxScaler
3. **build_save_model_task**: 
   - Tests K=1 to K=10
   - Calculates SSE for each K
   - Trains final KMeans model (K=5)
4. **load_model_elbow_task**: Uses elbow method to determine optimal clusters

## Dataset Information

**Mall Customers Dataset:**
- 200 customers
- Features used: Annual Income (k$), Spending Score (1-100)
- Use case: Customer segmentation for targeted marketing

## Setup Instructions

### Prerequisites

- Docker Desktop installed and running
- At least 4GB RAM allocated to Docker

### Installation
```bash
cd /Users/vaishnavisarmalkar/Documents/MLOps-Labs/Airflow_Lab

# Run setup script
./setup.sh

# Start Airflow
docker compose up
```

Wait 2-3 minutes for services to start.

### Access Airflow UI

1. Open: **http://localhost:8080**
2. Login: 
   - Username: `airflow`
   - Password: `airflow`

## Running the Pipeline

1. Find `mall_customer_clustering_dag` in the DAG list
2. Toggle switch to **enable** it
3. Click **Play button** -> **Trigger DAG**
4. Monitor execution in Graph view

**Expected runtime**: ~30-60 seconds

## Results

### Model Output

- **Saved model**: `model/customer_clustering_model.pkl`
- **Optimal clusters**: Determined by elbow method (typically 4-5)
- **Algorithm**: K-Means with k-means++ initialization

### Task Logs Show:
```
Loading Mall Customers dataset...
Dataset loaded: 200 customers, 5 features
Selected features for clustering: ['Annual Income (k$)', 'Spending Score (1-100)']
Building KMeans clustering model...
  k=1, SSE=269981.28
  k=2, SSE=183011.02
  k=3, SSE=106348.37
  k=4, SSE=44448.45
  k=5, SSE=31370.76
  ...
Optimal number of clusters: 5
Model saved to: /opt/airflow/model/customer_clustering_model.pkl
```

### Feature Selection
- Annual Income (k$): Customer purchasing power
- Spending Score (1-100): Customer spending behavior

### Data Preprocessing
- MinMaxScaler normalization (0-1 range)
- No outlier removal (small dataset)

### Model Selection
- K-Means with k-means++ initialization
- Elbow method for optimal K selection
- Range tested: K=1 to K=10

### XCom Data Passing
- Base64 encoding for binary pickle data
- Enables passing DataFrames between tasks

## Customer Segments (Typical Results)

The model typically identifies 5 segments:

1. **High Income, High Spending**: Premium customers
2. **High Income, Low Spending**: Potential upsell targets
3. **Medium Income, Medium Spending**: Regular customers
4. **Low Income, High Spending**: Loyal budget shoppers
5. **Low Income, Low Spending**: Occasional shoppers

## Stopping Airflow
```bash
# Stop services
docker compose down

# Stop and remove all data
docker compose down -v
```

