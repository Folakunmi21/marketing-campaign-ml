# Marketing-Campaign-Response-Predictor
This project builds a machine learning model that predicts whether a customer will respond to a marketing offer. It includes data exploration, model training, API development with FastAPI, containerization with Docker, and cloud deployment using Fly.io.

## Project Overview
Businesses often waste marketing budget by sending offers to customers who are unlikely to respond. This project solves that by predicting response likelihood using customer demographics, purchase patterns, and past campaign interactions.

## Dataset
The project uses the [Customer Marketing Campaign dataset]([url](https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign)) containing customer demographics, purchase behaviour, and previous campaign acceptance.

## Key Features
- End-to-end ML pipeline
- EDA + feature engineering
- Model training using Linear Regression, Random Forest, and XGBoost, with hyperparameter tuning and selection of the best-performing model (XGBoost)
- FastAPI web service (/response endpoint)
- Docker-ready application
- Cloud deployment support (Fly.io)

## Installation
**1. Clone the repository**
git clone <your-repo-url>
cd marketing-campaign-ml

**2. Create & activate environment**
uv venv
source .venv/bin/activate

**3. Install dependencies**
uv pip install -r requirements.txt

or (if using pyproject)

uv pip install .

## Local Development
**Run the API**
uvicorn response:app --host 0.0.0.0 --port 9696 --reload

**Test the API**
python test.py

You can also explore the interactive docs at:
http://localhost:9696/docs

## Docker Usage
**Build the image**
docker build -t response-prediction .

## Run the container
docker run -p 9696:9696 response-prediction

### Prediction Endpoint
POST /response

### Request body
Customer data (validated via Pydantic).


## Technologies Used
- Python 3.12+
- FastAPI
- XGBoost
- Scikit-learn
- Pydantic
- Docker
- Uvicorn
- Fly.io
