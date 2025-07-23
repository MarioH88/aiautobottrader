# Use official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy all project files
COPY . .

# Default command to run your Streamlit dashboard
CMD ["streamlit", "run", "dashboard_app.py", "--server.port=8080", "--server.address=0.0.0.0"]