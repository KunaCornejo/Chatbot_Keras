# Use an official Python runtime as a parent image
FROM python:3.11-slim
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
COPY ./src /app/src
COPY ./models /app/models
COPY ./data /app/data
# Install the required packages
RUN apt-get update && \
    apt-get install --no-install-recommends -qy python3-dev g++ gcc && \
    python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir --upgrade -r /app/requirements.txt
RUN useradd nonroot
USER nonroot
EXPOSE 8080
CMD ["python", "src/app.py"]