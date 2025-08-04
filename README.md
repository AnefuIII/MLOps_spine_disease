# MLOps Spine Disease Project

This repository contains my submission for the [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) course by DataTalks.Club. The project focuses on building a complete MLOps pipeline for spine disease classification using machine learning.

## Data Source

The dataset used in this project is the **Vertebral Column Dataset** from Kaggle:
- **Source**: [Vertebral Column Dataset](https://www.kaggle.com/datasets/jessanrod3/vertebralcolumndataset?resource=download)
- **Description**: Contains biomechanical features of the vertebral column for spine disease classification
- **Features**: 6 biomechanical attributes derived from the shape and orientation of the pelvis and lumbar spine
- **Target**: Classification of spine disease (Normal, Disk Hernia, Spondylolisthesis)

## Course Information

This project is a submission for the **MLOps Zoomcamp** course:
- **Course**: [MLOps Zoomcamp by DataTalks.Club](https://github.com/DataTalksClub/mlops-zoomcamp)
- **Duration**: 9-week course on productionizing ML services
- **Topics Covered**: Experiment tracking, orchestration, deployment, monitoring, and best practices

## Project Structure

The repository is organized into modules corresponding to the MLOps Zoomcamp curriculum:

- **01_mlopszc_EDA_model_training/**: Initial data exploration, model training, and experiment tracking with MLflow
- **mlopszc_proj_orc/**: Workflow orchestration using Mage AI for pipeline automation
- **03_mlopszc_web_deployment/**: Model deployment as a web service using Flask and Docker
- **04_mlopszc_monitoring/**: Model monitoring and drift detection using Evidently and Grafana

## Getting Started

Each folder contains its own README with specific setup and usage instructions. Start with the first module and work through each one sequentially.

### ⚠️ **Important: Virtual Environment Isolation**

**Each module should run in its own virtual environment** due to conflicting dependencies across different MLOps tools and frameworks:

- **01_mlopszc_EDA_model_training/**: MLflow, scikit-learn, pandas, numpy
- **mlopszc_proj_orc/**: Mage AI, specific orchestration dependencies  
- **03_mlopszc_web_deployment/**: Flask, pipenv, Docker dependencies
- **04_mlopszc_monitoring/**: Evidently, Prometheus, Grafana dependencies
- **05_mlopszc_best_practices/**: Testing, linting, CI/CD tools

### Setup Workflow

```bash
# For each module, create isolated environment:
cd module_directory/
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows
pip install -r requirements.txt
# Follow module-specific instructions...
```

## Requirements

- Python 3.8+
- Docker
- Virtual environment management (venv, conda, or similar)
- Basic knowledge of machine learning and Python programming

## License

This project is for educational purposes as part of the MLOps Zoomcamp course. 