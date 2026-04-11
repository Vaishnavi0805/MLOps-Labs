# Customer Sentiment Analysis API - Google Cloud Run Deployment

This lab demonstrates deploying a containerized sentiment analysis API to Google Cloud Run. The application analyzes customer reviews and classifies them as positive, negative, or neutral.

## Overview

This lab implements a real-world sentiment analysis API using Flask and TextBlob, deployed as a serverless container on Google Cloud Run.

## Project Structure
Cloud_Run_Lab/
├── app/
│   └── app.py              # Flask application
├── Dockerfile              # Container configuration
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md


## Application Features

### Endpoints

1. **GET /** - API information and available endpoints
2. **GET /health** - Health check endpoint
3. **POST /analyze** - Sentiment analysis endpoint

### Sentiment Analysis

The API uses TextBlob to analyze text and returns:
- **Sentiment**: positive, negative, or neutral
- **Polarity**: -1 (most negative) to 1 (most positive)
- **Subjectivity**: 0 (objective) to 1 (subjective)

## Prerequisites

- Python 3.11+
- Docker Desktop installed and running
- Google Cloud account with billing enabled
- Google Cloud SDK (`gcloud`) installed

## Local Testing

### Step 1: Install Dependencies

```bash
cd /Users/vaishnavisarmalkar/Documents/MLOps-Labs/Cloud_Run_Lab
source ../venv/bin/activate
pip install -r requirements.txt
python -m textblob.download_corpora
```

### Step 2: Run Flask App Locally

```bash
cd app
PORT=5050 python app.py
```

### Step 3: Test the API

```bash
# Home endpoint
curl http://localhost:5050/

# Health check
curl http://localhost:5050/health

# Positive sentiment
curl -X POST http://localhost:5050/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing and I love it!"}'

# Negative sentiment
curl -X POST http://localhost:5050/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Terrible service, very disappointing"}'
```

## Docker Deployment

### Step 1: Build Docker Image

```bash
cd /Users/vaishnavisarmalkar/Documents/MLOps-Labs/Cloud_Run_Lab
docker build -t sentiment-analysis-api .
```

### Step 2: Run Docker Container

```bash
docker run -p 5050:8080 sentiment-analysis-api
```

### Step 3: Test Docker Container

```bash
curl -X POST http://localhost:5050/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Great experience with this bank!"}'
```

## Google Cloud Run Deployment

### Step 1: Set Up Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (e.g., `sentiment-api-project`)
3. Note your PROJECT_ID

### Step 2: Enable Required APIs

```bash
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### Step 3: Authenticate with Google Cloud

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud auth configure-docker
```

### Step 4: Tag and Push Docker Image

```bash
# Tag the image
docker tag sentiment-analysis-api gcr.io/YOUR_PROJECT_ID/sentiment-analysis-api

# Push to Google Container Registry
docker push gcr.io/YOUR_PROJECT_ID/sentiment-analysis-api
```

### Step 5: Deploy to Cloud Run

```bash
gcloud run deploy sentiment-api \
  --image gcr.io/YOUR_PROJECT_ID/sentiment-analysis-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

This will:
- Deploy your container to Cloud Run
- Make it publicly accessible
- Auto-scale based on traffic
- Provide you with a public URL

### Step 6: Test the Deployed API

Cloud Run will provide a URL like: `https://sentiment-api-xxxxx-uc.a.run.app`

Test it:

```bash
# Replace with your actual Cloud Run URL
curl https://YOUR_CLOUD_RUN_URL/

curl -X POST https://YOUR_CLOUD_RUN_URL/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This banking app is user-friendly and efficient!"}'
```

## Monitoring and Scaling

### Monitor in Cloud Console

1. Go to Cloud Run in Google Cloud Console
2. Click on your service
3. View metrics:
   - Request count
   - Request latency
   - Container instances
   - Memory usage

### Auto-Scaling

Cloud Run automatically scales based on:
- **Traffic**: Spins up instances when requests increase
- **Zero-scale**: Scales down to 0 when no traffic (saves costs)
- **Concurrency**: Each instance handles multiple requests
