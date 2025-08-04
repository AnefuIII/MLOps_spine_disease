# Module 4: Model Monitoring

This module implements comprehensive model monitoring and drift detection for the spine disease classification service.

## What's Inside

- **Data Drift Detection**: Monitoring feature distributions over time
- **Model Performance Monitoring**: Tracking prediction accuracy and metrics
- **Quality Monitoring**: Data quality checks and validation
- **Dashboard Visualization**: Grafana dashboards for monitoring metrics
- **Automated Alerts**: Alerting when drift or performance issues are detected

## Key Files

- `docker-compose.yml`: Multi-container setup for monitoring stack
- `evidently_metrics_calculation.py`: Script for calculating monitoring metrics
- `baseline_model_spine_data.ipynb`: Baseline model and monitoring setup
- `data_drift_suite.html`: Evidently drift detection reports
- `spine_data_quality_report.html`: Data quality assessment reports
- `requirements.txt`: Python dependencies for monitoring

## Monitoring Stack

- **Evidently**: Data drift and model performance monitoring
- **Prometheus**: Metrics collection and storage
- **Grafana**: Dashboard visualization
- **MongoDB**: Monitoring data storage

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the monitoring stack:
```bash
docker-compose up -d
```

3. Access monitoring dashboards:
- Grafana: http://localhost:3000
- Evidently reports: Generated HTML files

## Monitoring Components

### Data Drift Detection
- **Feature Drift**: Monitor changes in feature distributions
- **Target Drift**: Track changes in prediction targets
- **Data Quality**: Validate data integrity and completeness

### Model Performance
- **Prediction Accuracy**: Track model performance over time
- **Prediction Drift**: Monitor changes in prediction distributions
- **Business Metrics**: Custom business-specific monitoring

### Automated Alerts
- **Drift Alerts**: Notify when significant drift is detected
- **Performance Alerts**: Alert on performance degradation
- **Data Quality Alerts**: Flag data quality issues

## Usage

1. Run the baseline model notebook to set up monitoring
2. Start the monitoring stack with docker-compose
3. Generate monitoring reports:
```bash
python evidently_metrics_calculation.py
```

4. View dashboards and reports in the browser

## Benefits

- **Proactive Monitoring**: Detect issues before they impact users
- **Automated Alerts**: Reduce manual monitoring overhead
- **Historical Tracking**: Maintain performance history
- **Actionable Insights**: Clear reports for decision making 