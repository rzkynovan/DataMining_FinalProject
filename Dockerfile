FROM python:3.8-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# FIXED: Correct working directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL application files
COPY . .

# Debug: List files to verify copy
RUN echo "=== Files in /app ===" && ls -la
RUN echo "=== Python files ===" && ls -la *.py

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=app.py
ENV FLASK_ENV=development

# Create directories
RUN mkdir -p uploads models templates static

# Test import to catch errors early
RUN python3 -c "import model_utils; print('✅ model_utils imported successfully')" || echo "❌ Import failed"

# Expose port (match docker-compose)
EXPOSE 8899

# FIXED: Run directly with python (not flask run)
CMD ["python3", "app.py"]
