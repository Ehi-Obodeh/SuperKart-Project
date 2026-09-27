FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code and model artifacts
COPY . .

# Expose the Flask port
EXPOSE 7860

# Run the Flask app
CMD ["python", "app.py"]
