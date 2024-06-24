# Use a Debian-based Python image for better compatibility with TensorFlow
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Upgrade pip to avoid dependency resolution issues
RUN pip install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install Pillow

# Make port 3000 available to the world outside this container
EXPOSE 3000

# Run the application
CMD ["python", "./index.py"]
