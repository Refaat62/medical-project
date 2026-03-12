#  Base image 
FROM python:3.10-slim

#  System dependencies 
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

#  Working directory 
WORKDIR /app

#  Copy files 
COPY requirements.txt .
COPY main.py .
COPY model_lung.py .

#  Install Python dependencies 
RUN pip install --no-cache-dir -r requirements.txt

#  Create weights directory (will be filled at runtime from HuggingFace) 
RUN mkdir -p weights

#  Expose port 
EXPOSE 8000

#  Start server 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
