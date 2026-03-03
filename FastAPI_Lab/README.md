# Wine Classification API - FastAPI Lab

This lab demonstrates building a Machine Learning API using FastAPI to predict wine classifications based on chemical properties. The project showcases MLOps practices including model training, API development, and interactive documentation.

## Project Overview

This API uses a Random Forest Classifier trained on the Wine dataset to predict wine classes (0, 1, or 2) based on 13 chemical measurements. Unlike basic classification examples, this project focuses on real-world wine quality assessment using scientifically measured features.

### Features

- **Machine Learning Model**: Random Forest Classifier for wine classification
- **RESTful API**: FastAPI-based endpoints for predictions
- **Interactive Documentation**: Auto-generated Swagger UI
- **Data Validation**: Pydantic models for request/response validation
- **Model Persistence**: Joblib for model serialization

## Project Structure
```
FastAPI_Lab/
├── assets/                    # Documentation assets (screenshots, diagrams)
├── model/
│   └── wine_model.pkl        # Trained Random Forest model
├── src/
│   ├── data.py               # Data loading and preprocessing
│   ├── train.py              # Model training script
│   ├── predict.py            # Prediction logic
│   └── main.py               # FastAPI application
├── requirements.txt           # Python dependencies
└── README.md                 # Project documentation
```

## Dataset Information

**Wine Dataset** (from scikit-learn):
- **13 Features**: Chemical measurements of wines
  - Alcohol, Malic acid, Ash, Alcalinity of ash
  - Magnesium, Total phenols, Flavanoids
  - Nonflavanoid phenols, Proanthocyanins
  - Color intensity, Hue, OD280/OD315, Proline
- **3 Classes**: Wine cultivars (0, 1, 2)
- **178 Samples**: Balanced dataset

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone and Navigate
```bash
cd FastAPI_Lab
```

### Step 2: Activate Virtual Environment

**On macOS/Linux:**
```bash
source ../venv/bin/activate
```

**On Windows:**
```bash
..\venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Application

### Step 1: Train the Model

Navigate to the src directory and train the model:
```bash
cd src
python train.py
```

You should see: `Model trained and saved successfully!`

This creates `wine_model.pkl` in the `model/` directory.

### Step 2: Start the API Server
```bash
uvicorn main:app --reload
```

The server will start at: `http://127.0.0.1:8000`

### Step 3: Access Interactive Documentation

Open your browser and visit:
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

## API Endpoints

### 1. Health Check

**Endpoint**: `GET /`

**Description**: Simple health check to verify the API is running

**Response**:
```json
{
  "status": "healthy"
}
```

### 2. Wine Classification Prediction

**Endpoint**: `POST /predict`

**Description**: Predict wine class based on chemical measurements

**Request Body**:
```json
{
  "alcohol": 13.2,
  "malic_acid": 2.3,
  "ash": 2.4,
  "alcalinity_of_ash": 18.5,
  "magnesium": 110.0,
  "total_phenols": 2.8,
  "flavanoids": 3.1,
  "nonflavanoid_phenols": 0.28,
  "proanthocyanins": 2.0,
  "color_intensity": 5.6,
  "hue": 1.05,
  "od280_od315": 3.2,
  "proline": 1100.0
}
```

**Response**:
```json
{
  "response": 0
}
```

**Response Codes**:
- `200`: Successful prediction
- `422`: Validation error (invalid input data)
- `500`: Internal server error

## Testing the API

### Using Swagger UI

1. Go to http://127.0.0.1:8000/docs
2. Click on `POST /predict`
3. Click "Try it out"
4. Paste the example request body
5. Click "Execute"
6. View the prediction result

### Using cURL
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "alcohol": 13.2,
  "malic_acid": 2.3,
  "ash": 2.4,
  "alcalinity_of_ash": 18.5,
  "magnesium": 110.0,
  "total_phenols": 2.8,
  "flavanoids": 3.1,
  "nonflavanoid_phenols": 0.28,
  "proanthocyanins": 2.0,
  "color_intensity": 5.6,
  "hue": 1.05,
  "od280_od315": 3.2,
  "proline": 1100.0
}'
```

### Using Python requests
```python
import requests

url = "http://127.0.0.1:8000/predict"
data = {
    "alcohol": 13.2,
    "malic_acid": 2.3,
    "ash": 2.4,
    "alcalinity_of_ash": 18.5,
    "magnesium": 110.0,
    "total_phenols": 2.8,
    "flavanoids": 3.1,
    "nonflavanoid_phenols": 0.28,
    "proanthocyanins": 2.0,
    "color_intensity": 5.6,
    "hue": 1.05,
    "od280_od315": 3.2,
    "proline": 1100.0
}

response = requests.post(url, json=data)
print(response.json())  # {'response': 0}
```

## Understanding Wine Classes

The model predicts one of three wine cultivar classes:
- **Class 0**: First wine cultivar
- **Class 1**: Second wine cultivar
- **Class 2**: Third wine cultivar

Each class represents wines from different grape cultivars grown in the same region of Italy.

## Technical Details

### Model Architecture

- **Algorithm**: Random Forest Classifier
- **Parameters**:
  - n_estimators: 100
  - max_depth: 5
  - random_state: 42
- **Train/Test Split**: 70% / 30%

### Data Validation

Pydantic models ensure:
- All 13 features are provided
- All values are numeric (float)
- Data types are correct
- Automatic error messages for invalid inputs

### API Framework

- **FastAPI**: Modern, fast web framework
- **Uvicorn**: ASGI server for production
- **Automatic Documentation**: OpenAPI/Swagger
- **Type Hints**: Full Python type checking

