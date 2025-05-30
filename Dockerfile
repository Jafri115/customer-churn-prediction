# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app

# Create and set the working directory
WORKDIR $APP_HOME

# Install system dependencies (if any, e.g., for specific libraries like libgomp1 for LightGBM)
# RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy just the requirements file to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the image
COPY ./src $APP_HOME/src
COPY ./models $APP_HOME/models 
# Note: Data is not copied as it's assumed to be handled by the pipeline before model serving

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the application
# Using uvicorn directly for simplicity, Gunicorn is better for production
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]