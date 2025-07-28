# Adobe Hackathon Challenge 1B: Persona-Driven PDF Analysis
FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (vendor all packages)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy Challenge 1A extractor (dependency)
COPY ../Challenge_1a/process_pdfs.py /app/challenge1a/

# Copy Challenge 1B application files
COPY analyze_collections.py .

# Create mount points
RUN mkdir -p /app/collections

# Run the analysis script
CMD ["python", "analyze_collections.py"]
