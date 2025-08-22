# 1. Base image: use official lightweight Python
FROM python:3.11-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy project files into image
COPY . .

# 5. Expose Flask's default port
EXPOSE 5000

# 6. Set environment variables for production (override via docker run)
ENV FLASK_APP=src.app
ENV FLASK_ENV=production

# 7. (Optional) Create data and models directory, if not present
RUN mkdir -p data models

# 8. Start the app
CMD ["python", "-m", "src.app"]
