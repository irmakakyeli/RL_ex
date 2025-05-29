FROM python:3.11.1
LABEL authors="irmak"

# Set working directory
WORKDIR /

# Install git (needed for cloning in model.py)
RUN apt-get update && apt-get install -y git

# Copy application files
COPY requirements.txt requirements.txt
COPY src/model.py /src/model.py

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run the Python script on container start
CMD ["python", "/src/model.py"]