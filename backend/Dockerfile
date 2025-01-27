# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy dependencies
COPY requirements.txt .
COPY model/fetch_model.py ./model/
COPY backend/ ./backend/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the model fetch script
RUN python model/fetch_model.py

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "app.py"]