# Use Python 3.9 slim image for smaller base size
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for NLP libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libc6-dev \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Python dependencies in order of size/importance
# Core dependencies first
RUN pip install --no-cache-dir \
    PyMuPDF==1.23.14 \
    && pip cache purge

# Scientific computing stack
RUN pip install --no-cache-dir \
    numpy==1.24.4 \
    scikit-learn==1.3.2 \
    && pip cache purge

# NLP dependencies
RUN pip install --no-cache-dir \
    nltk==3.8.1 \
    && pip cache purge

# Install spaCy and small English model (lightweight but powerful)
RUN pip install --no-cache-dir \
    spacy==3.7.2 \
    && pip cache purge

# Download spaCy model (only ~15MB)
RUN python -m spacy download en_core_web_sm

# Copy the application code
COPY challenge1b_main.py .

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Pre-download NLTK data to avoid runtime downloads and reduce startup time
RUN python -c "import nltk; \
nltk.download('punkt', quiet=True); \
nltk.download('stopwords', quiet=True); \
nltk.download('averaged_perceptron_tagger', quiet=True); \
nltk.download('maxent_ne_chunker', quiet=True); \
nltk.download('words', quiet=True); \
print('NLTK data downloaded successfully')"

# Set environment variables for optimal performance
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Optimize for CPU performance
ENV OPENBLAS_NUM_THREADS=4
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# Run the application
CMD ["python", "challenge1b_main.py"]