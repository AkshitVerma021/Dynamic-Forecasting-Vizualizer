# Use the official Python 3 image as base
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv

# Create and activate virtual environment
RUN python3 -m venv $VIRTUAL_ENV && \
    . $VIRTUAL_ENV/bin/activate && \
    pip install --upgrade pip

# Set PATH to include the virtual environment
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the application code to the container
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app when container starts
CMD ["streamlit", "run", "main.py", "--server.address", "0.0.0.0"]
