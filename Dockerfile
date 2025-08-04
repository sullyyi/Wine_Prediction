# Use a slim OpenJDK 8 image
FROM openjdk:8-jdk-slim

# Install Python3 and pip
RUN apt-get update \
 && apt-get install -y python3 python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Install PySpark and scikit-learn
RUN pip3 install pyspark==3.4.1 scikit-learn

# Create app directory
WORKDIR /app

# Copy prediction script and the trained model
COPY wine_quality_predict.py .
COPY model ./model

# Default command runs the prediction script
ENTRYPOINT ["python3", "wine_quality_predict.py"]
