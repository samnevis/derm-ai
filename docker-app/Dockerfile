# Use an official lightweight Python image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files into the container
COPY . .

# Expose the Flask API port
EXPOSE 5000

# Run the Flask app
CMD ["python", "api.py"]
