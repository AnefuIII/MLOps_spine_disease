# Use a slim Python 3.12 base image
FROM python:3.12.3-slim

# Upgrade pip and install pipenv
RUN pip install -U pip
RUN pip install pipenv

# Set the working directory inside the container
WORKDIR /app

# Copy Pipfile and Pipfile.lock into the /app directory
COPY Pipfile Pipfile.lock ./

# Install dependencies using pipenv in a system-wide manner inside the container
RUN pipenv install --system --deploy --ignore-pipfile

# Copy predict.py separately
COPY predict.py ./

# Copy log_reg.pkl separately
COPY log_reg.pkl ./

# Expose the port your Flask app will run on
EXPOSE 9696

# Define the command to run your application using Gunicorn
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]