# Module 1: EDA and Model Training

This module covers the initial data exploration, model training, and experiment tracking for the spine disease classification project.

## What's Inside

- **Data Exploration**: Jupyter notebooks for analyzing the vertebral column dataset
- **Model Training**: XGBoost and other ML models for spine disease classification
- **Experiment Tracking**: MLflow integration for tracking experiments and model versions
- **Model Persistence**: Saving trained models for later use

## Key Files

- `vert.ipynb`: Main data exploration and analysis notebook
- `xgboost.ipynb`: XGBoost model training and evaluation
- `final_model.py`: Production-ready model script
- `requirements.txt`: Python dependencies
- `mlflow.db`: MLflow database with experiment tracking
- `MLmodel`: MLflow model artifact

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start MLflow tracking server:
```bash
mlflow ui
```

## Usage

1. Run the data exploration notebook: `vert.ipynb`
2. Train models using: `xgboost.ipynb`
3. Use the final model script: `python final_model.py`

## Model Performance

The trained XGBoost model achieves high accuracy for spine disease classification with proper cross-validation and hyperparameter tuning. 