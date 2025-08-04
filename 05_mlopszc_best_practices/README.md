# Module 5: Best Practices

This module implements comprehensive MLOps best practices including testing, linting, formatting, automation, and CI/CD pipelines.

## What's Inside

- **Unit Tests**: Comprehensive test suite for all components
- **Integration Tests**: End-to-end testing of the complete pipeline
- **Code Quality**: Linting with flake8 and formatting with black
- **Automation**: Makefile for common development tasks
- **Pre-commit Hooks**: Automated code quality checks before commits
- **CI/CD Pipeline**: GitHub Actions for automated testing and deployment
- **Infrastructure as Code**: Terraform configuration for cloud resources
- **Virtual Environment Management**: Isolated environments for dependency conflicts

## Project Structure

```
05_mlopszc_best_practices/
├── src/                    # Source code
│   ├── __init__.py
│   ├── data_processor.py   # Data processing utilities
│   ├── model_trainer.py    # Model training logic
│   └── predictor.py        # Prediction service
├── tests/                  # Test files
│   ├── __init__.py
│   ├── test_data_processor.py
│   ├── test_model_trainer.py
│   ├── test_predictor.py
│   └── test_integration.py
├── .github/                # GitHub Actions workflows
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
├── terraform/              # Infrastructure as Code
│   ├── main.tf
│   ├── variables.tf
│   └── outputs.tf
├── Makefile               # Automation tasks
├── requirements.txt       # Python dependencies
├── requirements-dev.txt   # Development dependencies
├── .pre-commit-config.yaml # Pre-commit hooks
├── pyproject.toml        # Project configuration
├── .flake8              # Linting configuration
└── README.md            # This file
```

## Virtual Environment Management

### ⚠️ **Important: Module Isolation**

This project is designed to work with **isolated virtual environments** for each module due to conflicting dependencies across different MLOps tools and frameworks.

### Module-Specific Environments

Each module in the MLOps project should run in its own virtual environment:

- **01_mlopszc_EDA_model_training/**: MLflow, scikit-learn, pandas, numpy
- **mlopszc_proj_orc/**: Mage AI, specific orchestration dependencies
- **03_mlopszc_web_deployment/**: Flask, pipenv, Docker dependencies
- **04_mlopszc_monitoring/**: Evidently, Prometheus, Grafana dependencies
- **05_mlopszc_best_practices/**: Testing, linting, CI/CD tools

### Setup with Virtual Environment

1. **Create isolated virtual environment**:
```bash
# Create virtual environment for this module
python -m venv .venv

# Activate virtual environment
# On Linux/Mac:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

2. **Install dependencies**:
```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

3. **Install pre-commit hooks**:
```bash
pre-commit install
```

4. **Run tests**:
```bash
make test
```

### Environment Management Commands

```bash
# Create new virtual environment
make env-create

# Activate virtual environment (instructions only)
make env-activate

# Clean up virtual environment
make clean-env

# Install dependencies in current environment
make install-dev
```

## Cross-Module Integration

### Testing Integration Across Modules

The integration tests in this module are designed to work with **external module outputs**:

```python
# Example: Testing integration with other modules
def test_integration_with_other_modules():
    """
    Test integration with outputs from other modules.
    This assumes other modules have been run in their respective environments.
    """
    # Load model from Module 1
    # model_path = "../01_mlopszc_EDA_model_training/models/best_model.pkl"
    
    # Test prediction service from Module 3
    # service_url = "http://localhost:9696"
    
    # Validate monitoring data from Module 4
    # monitoring_data = "../04_mlopszc_monitoring/data/"
```

### Module Dependencies

This module depends on outputs from other modules:

- **Module 1**: Trained models and MLflow artifacts
- **Module 2**: Orchestrated pipeline outputs
- **Module 3**: Deployed web service endpoints
- **Module 4**: Monitoring data and metrics

### Running Cross-Module Tests

```bash
# Run integration tests that depend on other modules
pytest tests/test_integration.py -m "integration"

# Run tests with external module data (if available)
pytest tests/ --external-data-path="../other_modules/"
```

## Available Commands

- `make test`: Run all tests
- `make lint`: Run linting checks
- `make format`: Format code with black
- `make clean`: Clean up generated files
- `make build`: Build the project
- `make deploy`: Deploy to production (requires setup)
- `make env-create`: Create virtual environment
- `make env-activate`: Show activation instructions
- `make clean-env`: Clean virtual environment

## Testing

### Unit Tests
- Test individual components in isolation
- Mock external dependencies
- Fast execution for development
- **No dependency on other modules**

### Integration Tests
- Test complete workflows end-to-end
- Use test data and temporary resources
- Validate system behavior
- **May require outputs from other modules**

### Cross-Module Testing Strategy

```bash
# Test this module in isolation
make test-unit

# Test integration with other modules (if available)
make test-integration

# Test complete pipeline (requires all modules)
make test-pipeline
```

## Code Quality

- **Black**: Code formatting
- **Flake8**: Linting and style checking
- **Pre-commit**: Automated checks before commits
- **Type hints**: Python type annotations
- **Isolated environments**: Prevents dependency conflicts

## CI/CD Pipeline

### GitHub Actions Workflows
- **CI**: Automated testing on pull requests
- **CD**: Automated deployment on main branch
- **Security**: Dependency scanning and security checks
- **Environment isolation**: Each job runs in clean environment

### Infrastructure as Code
- **Terraform**: Cloud resource management
- **Modular design**: Reusable infrastructure components
- **Environment separation**: Dev, staging, production
- **Container isolation**: Docker for dependency management

## Best Practices Implemented

✅ **Unit Tests**: Comprehensive test coverage  
✅ **Integration Tests**: End-to-end workflow testing  
✅ **Linting & Formatting**: Code quality automation  
✅ **Makefile**: Development task automation  
✅ **Pre-commit Hooks**: Code quality gates  
✅ **CI/CD Pipeline**: Automated testing and deployment  
✅ **Infrastructure as Code**: Terraform for cloud resources  
✅ **Virtual Environment Management**: Isolated dependency management  

## Usage Examples

```bash
# Set up isolated environment
make env-create
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install dependencies
make install-dev

# Run all quality checks
make quality

# Run tests with coverage
make test-coverage

# Format and lint code
make format lint

# Deploy to staging
make deploy-staging
```

## Module Integration Workflow

### Complete Project Setup

1. **Set up each module in isolation**:
```bash
# Module 1: EDA and Model Training
cd 01_mlopszc_EDA_model_training/
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Run module 1...

# Module 2: Orchestration
cd ../mlopszc_proj_orc/
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Run module 2...

# Module 3: Web Deployment
cd ../03_mlopszc_web_deployment/
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Run module 3...

# Module 4: Monitoring
cd ../04_mlopszc_monitoring/
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Run module 4...

# Module 5: Best Practices (this module)
cd ../05_mlopszc_best_practices/
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
make test
```

2. **Run integration tests**:
```bash
# In Module 5 environment
make test-integration
```

## Benefits

- **Consistency**: Automated code quality standards
- **Reliability**: Comprehensive testing prevents regressions
- **Automation**: Reduced manual tasks and human error
- **Scalability**: Infrastructure as code for easy scaling
- **Collaboration**: Standardized development workflow
- **Isolation**: Prevents dependency conflicts between modules
- **Reproducibility**: Clear environment setup for each module 