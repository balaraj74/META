# Use Python 3.11 as the base image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
# Note: we use a combined requirement set
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the triage package and spaces code
# We expect the build context to have these
COPY . .

# Expose the port Gradio runs on
EXPOSE 7860

# Set environment variables
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860
ENV PYTHONUNBUFFERED=1

# Command to run the app
CMD ["python", "app.py"]
