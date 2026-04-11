# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Copy application files
COPY app/ /app/
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download TextBlob corpora (required for sentiment analysis)
RUN python -m textblob.download_corpora

# Expose port 8080 (Cloud Run default)
EXPOSE 8080

# Set environment variable for Flask
ENV FLASK_APP=app.py

# Run the application with gunicorn for production
CMD exec gunicorn --bind :8080 --workers 1 --threads 8 --timeout 0 app:app