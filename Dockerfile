# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install system dependencies (git for pip + build tools for JAX)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main application file to the working directory
COPY main.py .

# Make port 10000 available to the world outside this container
EXPOSE 10000

# Define environment variable
ENV NAME World

# Run main.py when the container launches
CMD ["python", "main.py"]
