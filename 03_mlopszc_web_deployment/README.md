# Module 3: Web Deployment

This module covers deploying the spine disease classification model as a web service using Flask and Docker.

## What's Inside

- **Flask Web Service**: REST API for spine disease prediction
- **Docker Containerization**: Containerized deployment for easy scaling
- **Model Serving**: Pre-trained models served via HTTP endpoints
- **Testing**: Unit tests for the web service

## Key Files

- `predict.py`: Main Flask application with prediction endpoints
- `test.py`: Unit tests for the web service
- `Dockerfile`: Docker configuration for containerization
- `Pipfile` & `Pipfile.lock`: Python dependencies using pipenv
- `xgbclassifier.pkl`: Trained XGBoost model
- `log_reg.pkl`: Trained Logistic Regression model

## Setup

1. Install dependencies using pipenv:
```bash
pipenv install
```

2. Activate the virtual environment:
```bash
pipenv shell
```

3. Run the Flask application:
```bash
python predict.py
```

## Docker Deployment

1. Build the Docker image:
```bash
docker build -t spine-disease-api .
```

2. Run the container:
```bash
docker run -p 9696:9696 spine-disease-api
```

## API Endpoints

- **POST /predict**: Make spine disease predictions
  - Input: JSON with biomechanical features
  - Output: Predicted spine disease class and probability

## Example Usage

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"pelvic_incidence": 63.0, "pelvic_tilt": 22.0, "lumbar_lordosis_angle": 39.0, "sacral_slope": 40.0, "pelvic_radius": 98.0, "degree_spondylolisthesis": 0.0}'
```

## Testing

Run the unit tests:
```bash
python test.py
```

## Production Considerations

- **Health Checks**: Built-in health check endpoint
- **Error Handling**: Proper error responses for invalid inputs
- **Logging**: Request and prediction logging
- **Scalability**: Docker containerization for easy scaling 