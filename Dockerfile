# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the application code into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "model_serving_api:app", "--host", "0.0.0.0", "--port", "8000"] 

# run               --> docker build -t fastapi-ml-predictor .   # bulid docker image 
#                   --> docker run -d -p 8000:8000 zereay/fastapi-ml-predictor  # run docker container
#                   --> docker login
# tag  to docker hub --> docker tag fastapi-ml-predictor:latest zereay/fastapi-ml-predictor:latest 
# push               --> docker push zereay/fastapi-ml-predictor:latest 
# connect RENDER or AWS to Docker hub for deployment.
