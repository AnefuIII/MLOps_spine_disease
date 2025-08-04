# Module 2: Workflow Orchestration

This module demonstrates workflow orchestration using Mage AI for automating the ML pipeline.

## What's Inside

- **Mage AI Pipeline**: Automated data processing and model training pipeline
- **Pipeline Components**: Data loading, preprocessing, model training, and evaluation
- **Scheduled Execution**: Automated pipeline runs for continuous model updates

## Key Files

- `spine_mage_pipeline/`: Main pipeline directory containing:
  - `spine_pipeline/`: Mage AI pipeline configuration
  - `data/`: Pipeline data storage
  - `mlruns/`: MLflow experiment tracking

## Setup

1. Install Mage AI:
```bash
pip install mage-ai
```

2. Start Mage AI:
```bash
mage start spine_mage_pipeline
```

3. Access the web interface at: `http://localhost:6789`

## Pipeline Components

The orchestrated pipeline includes:
- **Data Loading**: Reading the vertebral column dataset
- **Data Preprocessing**: Feature engineering and data cleaning
- **Model Training**: Automated model training with hyperparameter tuning
- **Model Evaluation**: Performance metrics calculation
- **Model Registration**: Saving models to MLflow registry

## Usage

1. Open Mage AI web interface
2. Navigate to the spine disease pipeline
3. Trigger manual runs or set up scheduled execution
4. Monitor pipeline performance and model metrics

## Benefits

- **Automation**: Reduces manual intervention in model training
- **Reproducibility**: Consistent pipeline execution
- **Scalability**: Easy to extend with new data sources or models
- **Monitoring**: Built-in pipeline monitoring and alerting 